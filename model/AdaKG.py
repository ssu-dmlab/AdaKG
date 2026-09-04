import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch_scatter import scatter_mean


class AdaKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.perturb_eps = config["perturb_eps"]
        self.cl_rate = config["cl_rate"]
        self.tau = config["tau"]
        self.require_pow = config["require_pow"]
        self.instability_metric = config["instability_metric"]
        self.message_drop = torch.nn.Dropout(config["mess_drop_rate"])

        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = torch.nn.Embedding(self.n_relations + 1, self.embedding_size)

        self.gamma_user = torch.nn.Parameter(torch.ones(1))
        self.gamma_item = torch.nn.Parameter(torch.ones(1))

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_user_e = None
        self.restore_item_e = None

        self.register_buffer("saved_user_alpha", torch.ones(self.n_users, 1))
        self.register_buffer("saved_item_alpha", torch.ones(self.n_items, 1))
        self.register_buffer("user_alpha_sum", torch.zeros(self.n_users, 1))
        self.register_buffer("user_alpha_count", torch.zeros(self.n_users, 1))
        self.register_buffer("item_alpha_sum", torch.zeros(self.n_items, 1))
        self.register_buffer("item_alpha_count", torch.zeros(self.n_items, 1))

        train_item_seen = np.zeros(self.n_items, dtype=bool)
        train_item_seen[self.interaction_matrix.col] = True

        self.register_buffer("train_item_seen",torch.from_numpy(train_item_seen).view(-1, 1))
        self.norm_adj_matrix, self.user_item_matrix = self.get_norm_adj()

        kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")

        self.head = torch.LongTensor(kg_graph.row).to(self.device)
        self.tail = torch.LongTensor(kg_graph.col).to(self.device)
        self.relation = torch.LongTensor(kg_graph.data).to(self.device)

        self.apply(xavier_uniform_initialization)

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),[1] * inter_M_t.nnz)
            )
        )
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        
        norm_adj_mat = sp.coo_matrix(D * A * D)
        norm_adj_matrix = self.to_sparse_tensor(norm_adj_mat).to(self.device)

        user_item_norm_mat = norm_adj_mat.tocsr()[:self.n_users, self.n_users:].tocoo()
        user_item_matrix = self.to_sparse_tensor(user_item_norm_mat).to(self.device)
        
        return norm_adj_matrix, user_item_matrix

    def to_sparse_tensor(self, sp_mat):
        sp_mat = sp_mat.tocoo()

        indices = torch.LongTensor(np.array([sp_mat.row, sp_mat.col]))
        values = torch.FloatTensor(sp_mat.data)

        sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(sp_mat.shape))
        return sparse_tensor
    
    def ig_forward(self, ig_embeddings):
        embeddings = ig_embeddings
        layer_embeddings = []

        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(self.norm_adj_matrix, embeddings)
            layer_embeddings.append(embeddings)


        return torch.stack(layer_embeddings, dim=1).mean(dim=1)

    def kg_layer(self, embeddings):
        entity_embeddings = embeddings[self.n_users:]
        relation_embeddings = self.relation_embedding.weight[self.relation]

        neighbor_embeddings = entity_embeddings[self.tail] * relation_embeddings

        user_embedding = torch.sparse.mm(self.user_item_matrix, entity_embeddings[:self.n_items])

        entity_embedding = scatter_mean(
            src=neighbor_embeddings,
            index=self.head,
            dim=0,
            dim_size=self.n_entities
        )
        entity_embedding = self.message_drop(entity_embedding)
        user_embedding = self.message_drop(user_embedding)
        
        entity_embedding = F.normalize(entity_embedding, dim=1)
        user_embedding = F.normalize(user_embedding, dim=1)
        
        kg_embedding = torch.cat([user_embedding, entity_embedding], dim=0)

        return kg_embedding

    def kg_forward(self, kg_embeddings):
        embeddings = kg_embeddings
        layer_embeddings = [embeddings]

        for _ in range(self.n_layers):
            embeddings = self.kg_layer(embeddings)
            layer_embeddings.append(embeddings)

        kg_final = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        return kg_final

    def compute_ig_instability(
        self,
        user,
        pos_item,
        neg_item,
        user_ig_emb,
        item_ig_emb,
        initial_ig_emb,
    ):
        pos_scores = (user_ig_emb[user] * item_ig_emb[pos_item]).sum(dim=1)
        neg_scores = (user_ig_emb[user] * item_ig_emb[neg_item]).sum(dim=1)
        margin = pos_scores - neg_scores

        grad = torch.autograd.grad(
            outputs=margin.mean(),
            inputs=initial_ig_emb,
            retain_graph=True,
            create_graph=False,
        )[0].detach()

        initial_emb_norm = initial_ig_emb.norm(dim=1, keepdim=True)
        delta = -self.perturb_eps * initial_emb_norm * F.normalize(grad, dim=1)
        perturbed_emb = (initial_ig_emb + delta).detach()

        z_tilde = self.ig_forward(perturbed_emb)
        z_clean = torch.cat([user_ig_emb, item_ig_emb],dim=0)

        if self.instability_metric == "cosine":
            cosine = F.cosine_similarity(z_clean, z_tilde, dim=1, eps=1e-8).unsqueeze(1)
            instability = 1.0 - cosine

        elif self.instability_metric == "symmetric_euclidean":
            diff_norm = (z_clean - z_tilde).norm(dim=1, keepdim=True)
            clean_norm = z_clean.norm(dim=1, keepdim=True)
            perturbed_norm = z_tilde.norm(dim=1, keepdim=True)

            instability = diff_norm / (clean_norm + perturbed_norm + 1e-8)

        batch_user_idx = torch.unique(user)
        batch_item_idx = torch.unique(torch.cat([pos_item, neg_item], dim=0))

        batch_item_idx = batch_item_idx[self.train_item_seen[batch_item_idx].squeeze(1)]
        batch_node_idx = torch.cat([batch_user_idx, self.n_users + batch_item_idx],dim=0)

        instability_centered = torch.zeros_like(instability)

        batch_instability = instability[batch_node_idx]
        batch_mean = batch_instability.mean()

        instability_centered[batch_node_idx] = batch_instability - batch_mean

        user_instability = instability_centered[:self.n_users]
        item_instability = instability_centered[self.n_users:]

        return (
            user_instability,
            item_instability,
            batch_user_idx,
            batch_item_idx
        )

    def compute_user_alpha(self, user_instability):
        gamma = F.softplus(self.gamma_user)
        return torch.sigmoid(gamma * user_instability)

    def compute_item_alpha(self, item_instability):
        gamma = F.softplus(self.gamma_item)
        return torch.sigmoid(gamma * item_instability)

    def reset_alpha_stats(self):
        self.user_alpha_sum.zero_()
        self.user_alpha_count.zero_()
        self.item_alpha_sum.zero_()
        self.item_alpha_count.zero_()

    def update_saved_alpha(self):
        user_mask = self.user_alpha_count.squeeze(1) > 0
        item_mask = self.item_alpha_count.squeeze(1) > 0

        self.saved_user_alpha[user_mask] = (
            self.user_alpha_sum[user_mask] / self.user_alpha_count[user_mask]
        )
        self.saved_item_alpha[item_mask] = (
            self.item_alpha_sum[item_mask] / self.item_alpha_count[item_mask]
        )
        
    def forward(self, user=None, pos_item=None, neg_item=None):
        initial_embeddings = torch.cat([self.user_embedding.weight, self.entity_embedding.weight], dim=0)
        ig_initial_emb = initial_embeddings[:self.n_users + self.n_items]

        ig_final = self.ig_forward(ig_initial_emb)
        kg_final = self.kg_forward(initial_embeddings)

        kg_final_ui = kg_final[:self.n_users + self.n_items]
        user_ig_emb = ig_final[:self.n_users]
        item_ig_emb = ig_final[self.n_users:]

        user_kg_emb = kg_final_ui[:self.n_users]
        item_kg_emb = kg_final_ui[self.n_users:]

        if self.training:
            user_instab, item_instab, batch_user_idx, batch_item_idx = self.compute_ig_instability(
                user, pos_item, neg_item, user_ig_emb, item_ig_emb, ig_initial_emb
            )
            user_alpha = self.saved_user_alpha.clone()
            item_alpha = self.saved_item_alpha.clone()

            user_alpha[batch_user_idx] = self.compute_user_alpha(user_instab[batch_user_idx])

            self.user_alpha_sum[batch_user_idx] += user_alpha[batch_user_idx].detach()
            self.user_alpha_count[batch_user_idx] += 1

            item_alpha[batch_item_idx] = self.compute_item_alpha(item_instab[batch_item_idx])
            pos_item_idx = torch.unique(pos_item)

            self.item_alpha_sum[pos_item_idx] += item_alpha[pos_item_idx].detach()
            self.item_alpha_count[pos_item_idx] += 1

            user_all_embeddings = user_ig_emb + user_alpha * user_kg_emb
            item_all_embeddings = item_ig_emb + item_alpha * item_kg_emb
    

            return (
                user_all_embeddings, item_all_embeddings,
                user_ig_emb, item_ig_emb, 
                user_kg_emb, item_kg_emb,
                user_alpha, item_alpha
            )

        user_all_embeddings = user_ig_emb + self.saved_user_alpha * user_kg_emb
        item_all_embeddings = item_ig_emb + self.saved_item_alpha * item_kg_emb

        return user_all_embeddings, item_all_embeddings

    def weighted_infonce(self, ig_emb, kg_emb, weight):
        ig_emb = F.normalize(ig_emb, dim=1)
        kg_emb = F.normalize(kg_emb, dim=1)

        pos = (ig_emb * kg_emb).sum(dim=1)
        sim = torch.matmul(ig_emb, kg_emb.t())

        pos = torch.exp(pos / self.tau)
        denom = torch.exp(sim / self.tau).sum(dim=1)

        cl_loss = -torch.log(pos / (denom + 1e-8))

        return (weight * cl_loss).mean()

    def adaptive_cl_loss(
        self,
        user,
        pos_item,
        user_ig_emb,
        item_ig_emb,
        user_kg_emb,
        item_kg_emb,
        user_alpha,
        item_alpha,
    ):
        user_idx = torch.unique(user)
        item_idx = torch.unique(pos_item)

        user_weight = user_alpha[user_idx].detach().squeeze(1)
        item_weight = item_alpha[item_idx].detach().squeeze(1)

        user_cl_loss = self.weighted_infonce(
            user_ig_emb[user_idx],
            user_kg_emb[user_idx],
            user_weight
        )

        item_cl_loss = self.weighted_infonce(
            item_ig_emb[item_idx],
            item_kg_emb[item_idx],
            item_weight
        )
        
        acl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)
        
        return acl_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        (
            user_all_embeddings, item_all_embeddings,
            user_ig_emb, item_ig_emb, 
            user_kg_emb, item_kg_emb,
            user_alpha, item_alpha
        ) = self.forward(user=user, pos_item=pos_item, neg_item=neg_item)

        u_emb = user_all_embeddings[user]
        pos_emb = item_all_embeddings[pos_item]
        neg_emb = item_all_embeddings[neg_item]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        reg_loss = self.reg_loss(
            self.user_embedding(user),
            self.entity_embedding(pos_item),
            self.entity_embedding(neg_item),
            require_pow=self.require_pow
        )

        cl_loss = self.adaptive_cl_loss(
            user, pos_item,
            user_ig_emb, item_ig_emb,
            user_kg_emb, item_kg_emb,
            user_alpha, item_alpha
        )
        total_loss = mf_loss + self.reg_weight * reg_loss + cl_loss
        
        return total_loss
        

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        scores = (user_all_embeddings[user] * item_all_embeddings[item]).sum(dim=1)

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_emb = self.restore_user_e[user]
        scores = torch.matmul(u_emb, self.restore_item_e.t())

        return scores.view(-1)
from recbole.trainer import KGTrainer


class AdaKGTrainer(KGTrainer):

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.reset_alpha_stats()

        result = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=loss_func,
            show_progress=show_progress,
        )

        self.model.update_saved_alpha()

        return result
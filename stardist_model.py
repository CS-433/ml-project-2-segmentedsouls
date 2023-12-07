from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D
import tqdm

class Stardist():
    def __init__(self, n_rays=32, use_gpu=False, grid=(2,2)):
        self.n_rays = n_rays
        self.use_gpu = use_gpu
        self.grid = grid
        self.score = list()
        self.taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.scores = list()

    def train(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, epochs, steps_per_epoch, cv=False):
        from tqdm.notebook import tqdm  # Use this if you're in a Jupyter notebook

        n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
        conf = Config2D(
            n_rays= self.n_rays,
            grid= self.grid,
            use_gpu= self.use_gpu,
            n_channel_in= n_channel,
            train_patch_size= (len(X_trn[0]), len(X_trn[0])),
            train_learning_rate= learning_rate
        )
        model = StarDist2D(conf, name='stardist', basedir='models')
        model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=epochs, steps_per_epoch=steps_per_epoch)

        model.optimize_thresholds(X_val, Y_val)
        Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(X_val)]

        stats = [matching_dataset(Y_val[i], Y_val_pred[i], thresh=t, show_progress=False) for i, t in enumerate(self.taus)]
        if cv:
            self.scores.append(
                {
                "learning_rate":learning_rate,
                "epochs":epochs,
                "steps_per_epoch":steps_per_epoch, 
                "stats":stats
                }
                )
        else:
            self.score = stats

    def CV(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epoch_settings, step_epochs, cv=True):
        for learning_rate in learning_rates:
            for epoch, steps_per_epoch in zip(epoch_settings, step_epochs):
                self.train(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, epoch, steps_per_epoch, cv)
        return self.scores
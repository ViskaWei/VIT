#region --PLOT-----------------------------------------------------------
import os
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde, skew
from src.utils import calculate_rms, calculate_snr, get_equivalent_width
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import pandas as pd

class SpecPlotter():
    def __init__(self, dataset):
        self.d = dataset
        self.wave = dataset.wave
        self.ca_rng = [8475, 8680]
        self.ca_mask = self.get_mask_from_wave_range(self.ca_rng)
        self.dpi = 100
        self.df_equivalent_width = None

    def get_mask_from_wave_range(self, wave_range):
        return (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])

    def plot_idx(self, i, lw=0.8):
        # Create figure with 1 row and 5 columns (left, middle-top, middle-bottom, right-top, right-bottom)
        fig = plt.figure(figsize=(16, 6), dpi=self.dpi)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        x, y, d = self.d.flux[i], self.d.noisy[i], self.d.denoised[i]
        
        snr0 = self.d.snr0[i] if hasattr(self.d, 'snr0') else calculate_rms(flux=x, noisy=y)
        snr = calculate_rms(flux=x, noisy=d)
        ca_snr0 = calculate_rms(flux=x[self.ca_mask], noisy=y[self.ca_mask])
        ca_snr = calculate_rms(flux=x[self.ca_mask], noisy=d[self.ca_mask])
        
        # Left plot (stays single)
        ax0 = fig.add_subplot(gs[:, 0])  # Takes all rows in first column
        ax0.plot(self.wave, y, c='k', alpha=1., label=f'y, S/N[y]={snr0:.0f}', lw=lw-0.2)
        ax0.plot(self.wave, x, c='r', lw=lw, label='x')
        ax0.axvspan(*self.ca_rng, color='orange', alpha=0.3, label='Ca II Region')
        ax0.set(xlim=(self.wave[0], self.wave[-1]), 
            xlabel='Wavelength (A)', 
            ylabel='Normalized Flux', 
            ylim=(-0., 1.))
        for line in ax0.legend().get_lines(): 
            line.set_linewidth(4)
        
        # Middle plots (top and bottom)
        ax1_top = fig.add_subplot(gs[0, 1])  # Top middle
        ax1_bottom = fig.add_subplot(gs[1, 1])  # Bottom middle
        offset = 0.2
        
        # Top middle plot (x)
        ax1_top.plot(self.wave, x, c='r', lw=lw, label='x')
        ax1_top.plot(self.wave, d-offset, c='b', lw=lw, label=f'd - {offset}')

        ax1_top.axvspan(*self.ca_rng, color='orange', alpha=0.3, label='Ca II Region')
        ax1_top.set(xlim=(self.wave[0], self.wave[-1]), 
                # ylim=(-0.12, 0.55),
                xticklabels=[])
        for line in ax1_top.legend(loc='lower center').get_lines()[:2]: 
            line.set_linewidth(4)
        # ax1_top.legend(loc='lower center').get_lines()[0].set_linewidth(4)
        
        # Bottom middle plot (x - d)
        ax1_bottom.plot(self.wave, x - d, c='teal', lw=lw, label=f'x - d, S/N[d]={snr:.0f}')

        ax1_bottom.axvspan(*self.ca_rng, color='orange', alpha=0.3, label='Ca II Region')
        ax1_bottom.set(xlim=(self.wave[0], self.wave[-1]), 
                    xlabel='Wavelength (A)',)
                    # ylim=(-0.12, 0.55))
        ax1_bottom.legend().get_lines()[0].set_linewidth(4)
        
        # Right plots (top and bottom)
        ax2_top = fig.add_subplot(gs[0, 2])  # Top right
        ax2_bottom = fig.add_subplot(gs[1, 2])  # Bottom right
        
        # Top right plot (x)
        ax2_top.plot(self.wave, x, c='r', lw=lw, label=f'x')
        ax2_top.plot(self.wave, d-offset, c='b', lw=lw, label=f'd - {offset}')
        ax2_top.axvspan(*self.ca_rng, color='orange', alpha=0.3, label='Ca II Region')
        ax2_top.set(xlim=(self.ca_rng[0]-10, self.ca_rng[-1]+10),
                # ylim=(-0.12, 0.55),
                xticklabels=[])
        for line in ax2_top.legend(loc='lower center').get_lines()[:2]: 
            line.set_linewidth(4)
        # ax2_top.legend(loc='lower center').get_lines()[0].set_linewidth(4)
        # ax2_top.legend(loc='lower center').get_lines()[1].set_linewidth(4)

        
        # Bottom right plot (x - d)
        ax2_bottom.plot(self.wave, x - d, c='teal', lw=lw, label=f'x - d, S/N[d_Ca]={ca_snr:.0f}')
        ax2_bottom.axvspan(*self.ca_rng, color='orange', alpha=0.3, label='Ca II Region')
        ax2_bottom.set(xlim=(self.ca_rng[0]-10, self.ca_rng[-1]+10),
                    xlabel='Wavelength (A)',)
                    # ylim=(-0.12, 0.55))
        ax2_bottom.legend(loc='upper center').get_lines()[0].set_linewidth(4)
        
        plt.tight_layout()
        return fig

    def plot_snr_improve(self, N=10000, num_levels= 20, start_idx=3, bnd = [1, 10]):
        snr0 = self.d.snr_no_mask[:N]
        snr1 = self.d.snr[:N] 
        snr01 = np.vstack([snr0, snr1])
        kde = gaussian_kde(snr01)
        z = kde(snr01)
        x_min, x_max = snr0.min().item(), snr0.max().item()
        y_min, y_max = snr1.min().item(), snr1.max().item()
        X, Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        levels = np.linspace(Z.min(), Z.max(), num_levels)
        mask = (z < levels[start_idx])

        fig, ax = plt.subplots(figsize=(5,4), dpi=self.dpi, facecolor='w')
        cs = ax.contourf(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # , norm=norm

        # ax.contour(X, Y, Z,
        #    levels=np.percentile(Z, [50, 70, 85, 95, 98, 99]),  # 少量主等高线
        #    colors='black', linewidths=0.8)
        # cs = ax.contour(X, Y, Z, levels=levels[start_idx:], cmap='hot',)  
        cbar = fig.colorbar(cs, ax=ax)
        cbar.locator = ticker.MaxNLocator(nbins=6)
        cbar.set_label("Density")
        ax.scatter(snr0[mask], snr1[mask], s=0.5, c='k',)
        _=ax.set(ylim=(0, None), xlabel="Initial (Noisy) S/N ", ylabel="Final (Denoised) S/N")       

        fig1, ax1 = plt.subplots(figsize=(5,4), dpi=self.dpi, facecolor='w')
        # cs = ax1.contour(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # 去掉最外圈
        cs1 = ax1.contourf(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # , norm=norm
        cbar1 = fig1.colorbar(cs1, ax=ax1)
        cbar1.locator = ticker.MaxNLocator(nbins=6)
        cbar1.set_label("Density")
        
        ax1.scatter(snr0[mask], snr1[mask], s=0.5, c='k')
        ax1.plot(bnd, bnd, 'k:', label='Noisy')
        major_ticks = [1, 2, 3, 4, 5, 10]  # 可以按需增加
        _=ax1.set(xlabel='Initial (Noisy) S/N (log scale)', xscale='log', yscale='log', ylim=(1,None), ylabel='Final (Denoised) S/N (log scale)', xticks=major_ticks, xticklabels=[str(m) for m in major_ticks])
        return fig, fig1

    def get_equivalent_width(self, N=100):
        z = self.d.z[:N]
        df_flux = get_equivalent_width(self.wave, self.d.flux[:N], z)
        df_noisy =  get_equivalent_width(self.wave, self.d.noisy[:N], z)
        df_denoised = get_equivalent_width(self.wave, self.d.denoised[:N], z)
        return df_flux, df_noisy, df_denoised


    def get_equivalent_width_error_dict(self, N=1000, bins=None):
        self.index_columns = ["TiO_4", "Ca1_LB13", "Ca2_LB13", "Ca3_LB13"]
        if bins is None: bins = np.array([0, 2, 4, 6, 8, np.inf])
        snr_values = self.d.snr_no_mask[:N]
        df_flux, df_noisy, df_denoised = self.get_equivalent_width(N=N)
        error_data = []
        self.eq_bins = []
        self.rel_err_dict = {index: {f"S/N < {int(self.eq_bins[i])}": None for i in range(len(self.eq_bins)-1)} for index in self.index_columns}
        self.rel_err_mean_dict = {index: {'noisy': [], 'denoised': []} for index in self.index_columns}
        self.rel_err_std_dict = {index: {'noisy': [], 'denoised': []} for index in self.index_columns}
        for index_name in self.index_columns:
            rel_err0 = (df_noisy[index_name] - df_flux[index_name]) / df_flux[index_name]
            rel_err1 = (df_denoised[index_name] - df_flux[index_name]) / df_flux[index_name]
            for i in range(len(bins) - 1):
                snr_min, snr_max = bins[i], bins[i+1]
                idx = np.where((snr_values >= snr_min) & (snr_values < snr_max))[0]
                data_noisy, data_denoised = rel_err0[idx], rel_err1[idx]
                if len(data_noisy) == 0: break
                self.eq_bins.append(snr_min)
                self.rel_err_dict[index_name][f"S/N < {int(bins[i+1])}"] = (data_noisy, data_denoised)
                noisy_mean = data_noisy.mean()
                denoised_mean = data_denoised.mean()
                noisy_std = data_noisy.std()
                denoised_std = data_denoised.std()

                self.rel_err_mean_dict[index_name]['noisy'].append(noisy_mean)
                self.rel_err_mean_dict[index_name]['denoised'].append(denoised_mean)
                self.rel_err_std_dict[index_name]['noisy'].append(noisy_std)
                self.rel_err_std_dict[index_name]['denoised'].append(denoised_std)

                for err in data_denoised:
                    error_data.append({'index_name': index_name, 'snr_bin': f"{int(bins[i])} - {int(bins[i+1])}", 'type': 'denoised', 'bias': err, 'mean': denoised_mean, 'std': denoised_std})
                for err in data_noisy:
                    error_data.append({'index_name': index_name, 'snr_bin': f"{int(bins[i])} - {int(bins[i+1])}", 'type': 'noisy', 'bias': err, 'mean': noisy_mean, 'std': noisy_std})
        self.df_equivalent_width = pd.DataFrame(error_data)
        self.eq_N = N
        self.eq_bins = self.eq_bins[:len(self.index_columns)]

    def plot_equivalent_width(self, plot_snr_bin='0 - 2', plot_name="Ca2_LB13"):
        if self.df_equivalent_width is None:
            self.get_equivalent_width_error_dict()
        noisy_data = self.df_equivalent_width[(self.df_equivalent_width['index_name'] == plot_name) & (self.df_equivalent_width['snr_bin'] == plot_snr_bin) & (self.df_equivalent_width['type'] == 'noisy')]['bias'].values
        denoised_data = self.df_equivalent_width[(self.df_equivalent_width['index_name'] == plot_name) & (self.df_equivalent_width['snr_bin'] == plot_snr_bin) & (self.df_equivalent_width['type'] == 'denoised')]['bias'].values
        
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        name = r'Ca II - 8542$\AA$'
        bin_name = f'S/N < {int(plot_snr_bin.split("-")[1].strip())}'
        SpecPlotter.plot_kde_distribution(ax, noisy_data, label="Noisy", color="k", bin_label=bin_name, index_name=name)
        SpecPlotter.plot_kde_distribution(ax, denoised_data, label="Denoised", color="blue", bin_label=bin_name, index_name=name)
        ax.axvline(0, color='r', lw=1, linestyle='-.')
        ax.set(xlabel="EW Relative Error Distribution", ylabel="Probability Density", xlim=(-2, 2))
        return fig

    def plot_equivalent_width_violin(self, plot_name="Ca2_LB13"):
        dff = self.df_equivalent_width[self.df_equivalent_width['index_name'] == plot_name].copy()
        dff['snr0_group1'] = dff['snr_bin']
        eq_stds_noisy = dff[dff['type'] == 'noisy'].groupby('snr_bin')['std'].first().to_dict()
        eq_stds_denoised = dff[dff['type'] == 'denoised'].groupby('snr_bin')['std'].first().to_dict()
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        ax.axvline(0, linestyle='-', color='r', alpha=1, lw=0.5)
        name = r'Ca II - 8542$\AA$'
        ax.text(0.5, 0.98, name, transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        sns.violinplot(x='bias', y='snr0_group1', hue='type', data=dff,split=True, gap=0.05, inner='quart', palette=['b','gray'], hue_order=['denoised', 'noisy'],linecolor='k', linewidth=0.5, ax=ax)
        ax.set(ylabel='Initial (Noisy) S/N Bin', xlabel='Equivalent Width Relative Error Distribution', xlim=(-1, 1))
        yticks = plt.gca().get_yticks()
        for i, y in enumerate(yticks[:5]):  # Assuming first 5 bins correspond
            bin_name = dff['snr_bin'].unique()[i]
            ax.text(-0.95, y - 0.08, f'σ = {eq_stds_denoised.get(bin_name, 0):.1g}', ha='left', va='center', fontsize=7, color='b')
            ax.text(-0.95, y + 0.12, f'σ = {eq_stds_noisy.get(bin_name, 0):.1g}', ha='left', va='center', fontsize=7, color='k')
        ax.legend(loc='upper right')
        return fig

    def plot_ew_std_comparison(self, index_columns=None):
        if index_columns is None: index_columns = self.index_columns
        bins = self.eq_bins
        color = ['k', 'b', 'r', 'g']
        marker = ['*', 'X', '^', 'o']
        fig, ax = plt.subplots(1, figsize=(5, 4), dpi=200)
        for i, index_name in enumerate(index_columns):
            std_denoised = self.rel_err_std_dict[index_name]['denoised']
            std_noisy = self.rel_err_std_dict[index_name]['noisy']
            ax.plot(bins, std_denoised, marker=marker[i], linestyle='-', c=color[i], label=f'{index_name[:3]} denoised', ms=5, lw=1)
            ax.plot(bins, std_noisy, marker=marker[i], linestyle=':', c=color[i], label=f'{index_name[:3]} noisy', ms=5, lw=1)

        ax.set(xlabel='Initial (Noisy) S/N', ylabel='Equivalent Width Relative Error SD (σ)', yscale='log', xticks=bins, xticklabels=[f'{m:.0f} - {m+2:.0f}' for m in bins])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,frameon=True, ncol=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig


    @staticmethod
    def add_stats_text(ax, data, label="Denoised"):
        """Add mean, std, skew stats box to ax."""
        mean_val, std_val, skew_val = np.mean(data), np.std(data), skew(data)
        stats_text = (f'{label}:\n μ={mean_val:.1g}\n σ={std_val:.1g}\n skew={skew_val:.1g}')
        x_pos = 0.95
        y_pos = 0.95 if label == 'Denoised' else 0.7
        color = 'b' if label == 'Denoised' else 'k'
        ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=10,color=color,
                bbox=dict(boxstyle='round', edgecolor=color, facecolor='white', alpha=0.8))

    @staticmethod
    def plot_kde_distribution(ax, data, label="", color="blue", bin_label="", index_name=""):
        data = np.asarray(data, dtype=np.float64)
        data = data[~np.isnan(data)]
        kde = gaussian_kde(data)
        x_vals = np.linspace(np.quantile(data, 0.001), np.quantile(data,0.999), 200)
        pdf_vals = kde(x_vals)
        ax.plot(x_vals, pdf_vals, c=color, label=label)
        ax.hist(data, bins=50, density=True, alpha=0.2, color=color)
        SpecPlotter.add_stats_text(ax, data, label=label)
        ax.text(0.05, 0.95, index_name, transform=plt.gca().transAxes, ha='left', va='top', fontsize=12)
        ax.text(0.05, 0.85, bin_label, transform=plt.gca().transAxes, ha='left', va='top', fontsize=12)


#region --REGRESSION PLOTTER-----------------------------------------------------------
class RegressionPlotter:
    """
    Comprehensive plotter for regression model evaluation.
    Supports multi-output regression (e.g., Teff, log_g, M_H).
    """
    
    def __init__(self, predictions, labels, param_names=None, logger=None, save_dir='./results/test_plots'):
        """
        Args:
            predictions: np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            labels: np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            param_names: List of parameter names (e.g., ['Teff', 'log_g', 'M_H'])
            logger: Optional W&B logger
            save_dir: Directory to save plots
        """
        import os
        self.predictions = np.asarray(predictions)
        self.labels = np.asarray(labels)
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine if multi-output
        self.is_multi_output = len(self.predictions.shape) > 1 and self.predictions.shape[1] > 1
        
        if self.is_multi_output:
            self.n_outputs = self.predictions.shape[1]
            self.param_names = param_names or ['Teff', 'log_g', 'M_H'][:self.n_outputs]
        else:
            self.n_outputs = 1
            self.param_names = param_names or ['Parameter']
            # Ensure predictions and labels are 2D for consistency
            if len(self.predictions.shape) == 1:
                self.predictions = self.predictions.reshape(-1, 1)
                self.labels = self.labels.reshape(-1, 1)
    
    def _save_and_log(self, fig, name):
        """Save figure locally and log to W&B if available"""
        # Save locally
        filename = name.replace('/', '_').replace('\\', '_') + '.png'
        filepath = os.path.join(self.save_dir, filename)
        try:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save figure {filename}: {e}")
        
        # Log to W&B
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                import wandb
                self.logger.experiment.log({name: wandb.Image(fig)})
            except Exception as e:
                print(f"Warning: Could not log figure to W&B: {e}")
        
        return fig
    
    def plot_predictions_vs_true(self):
        """Plot 1: Scatter plots of predictions vs true values"""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            preds_i = self.predictions[:, i]
            labels_i = self.labels[:, i]
            
            ax.scatter(labels_i, preds_i, alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val, max_val = labels_i.min(), labels_i.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Calculate metrics
            residuals = preds_i - labels_i
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((labels_i - labels_i.mean())**2)
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/predictions_vs_true')
    
    def plot_residual_distributions(self):
        """Plot 2: Residual distribution histograms"""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 4))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            residuals = self.predictions[:, i] - self.labels[:, i]
            
            ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero error')
            ax.axvline(np.median(residuals), color='g', linestyle='--', lw=2, 
                      label=f'Median={np.median(residuals):.4f}')
            
            ax.set_xlabel(f'Residual (Pred - True) for {name}')
            ax.set_ylabel('Count')
            ax.set_title(f'{name} Residual Distribution\nStd={np.std(residuals):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/residual_distributions')
    
    def plot_metrics_comparison(self):
        """Plot 3: Bar chart comparing metrics across parameters"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics_data = {'MAE': [], 'RMSE': [], 'R²': []}
        for i in range(self.n_outputs):
            residuals = self.predictions[:, i] - self.labels[:, i]
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((self.labels[:, i] - self.labels[:, i].mean())**2)
            
            metrics_data['MAE'].append(mae)
            metrics_data['RMSE'].append(rmse)
            metrics_data['R²'].append(r2)
        
        x = np.arange(self.n_outputs)
        width = 0.25
        
        ax.bar(x - width, metrics_data['MAE'], width, label='MAE', alpha=0.8)
        ax.bar(x, metrics_data['RMSE'], width, label='RMSE', alpha=0.8)
        ax.bar(x + width, metrics_data['R²'], width, label='R²', alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Metrics per Parameter')
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/metrics_per_parameter')
    
    def plot_residual_correlation(self):
        """Plot 4: Correlation heatmap of residuals (multi-output only)"""
        if self.n_outputs < 2:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        residuals_all = self.predictions - self.labels
        corr_matrix = np.corrcoef(residuals_all.T)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(self.n_outputs))
        ax.set_yticks(np.arange(self.n_outputs))
        ax.set_xticklabels(self.param_names)
        ax.set_yticklabels(self.param_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        # Add correlation values
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('Residual Correlation Heatmap')
        plt.tight_layout()
        return self._save_and_log(fig, 'test/residual_correlation')
    
    def plot_error_vs_true(self):
        """Plot 5: Error vs true value (detect systematic biases)"""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            residuals = self.predictions[:, i] - self.labels[:, i]
            labels_i = self.labels[:, i]
            
            ax.scatter(labels_i, residuals, alpha=0.5, s=10)
            ax.axhline(0, color='r', linestyle='--', lw=2, label='Zero error')
            
            # Add moving average to show trends
            sorted_idx = np.argsort(labels_i)
            window = max(len(labels_i) // 20, 3)  # Ensure minimum window size
            if window > 1 and len(labels_i) >= window:
                moving_avg = np.convolve(residuals[sorted_idx], 
                                        np.ones(window)/window, mode='valid')
                # Calculate correct indices for moving average
                start_idx = window // 2
                moving_x = labels_i[sorted_idx][start_idx:start_idx + len(moving_avg)]
                ax.plot(moving_x, moving_avg, 'g-', lw=2, label='Moving average')
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Residual (Pred - True)')
            ax.set_title(f'{name}: Error vs True Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/error_vs_true')
    
    def plot_qq(self):
        """Plot 6: Q-Q plots for residual normality check"""
        try:
            from scipy import stats
        except ImportError:
            print("Warning: scipy not available, skipping Q-Q plots")
            return None
        
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            residuals = self.predictions[:, i] - self.labels[:, i]
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{name}: Q-Q Plot\n(Check residual normality)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/qq_plots')
    
    def plot_comprehensive_summary(self):
        """Plot 10: Comprehensive 3-row summary figure"""
        from scipy.stats import norm
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, self.n_outputs, hspace=0.3, wspace=0.3)
        
        for i, name in enumerate(self.param_names):
            preds_i = self.predictions[:, i]
            labels_i = self.labels[:, i]
            residuals = preds_i - labels_i
            
            # Row 1: Scatter plot with metrics
            ax1 = fig.add_subplot(gs[0, i])
            ax1.scatter(labels_i, preds_i, alpha=0.4, s=5)
            min_val, max_val = labels_i.min(), labels_i.max()
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((labels_i - labels_i.mean())**2)
            
            ax1.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax1.set_xlabel(f'True {name}')
            ax1.set_ylabel(f'Predicted {name}')
            ax1.set_title(f'{name}')
            ax1.grid(True, alpha=0.3)
            
            # Row 2: Residual distribution
            ax2 = fig.add_subplot(gs[1, i])
            ax2.hist(residuals, bins=40, alpha=0.7, edgecolor='black', density=True)
            
            # Fit normal distribution
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax2.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal fit\nμ={mu:.4f}\nσ={sigma:.4f}')
            ax2.axvline(0, color='g', linestyle='--', lw=2, label='Zero')
            ax2.set_xlabel('Residual')
            ax2.set_ylabel('Density')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Row 3: Error vs True value
            ax3 = fig.add_subplot(gs[2, i])
            ax3.scatter(labels_i, residuals, alpha=0.4, s=5)
            ax3.axhline(0, color='r', linestyle='--', lw=2)
            
            # Add standard deviation bands
            ax3.axhline(sigma, color='orange', linestyle=':', lw=1.5, label=f'±1σ')
            ax3.axhline(-sigma, color='orange', linestyle=':', lw=1.5)
            ax3.axhline(2*sigma, color='red', linestyle=':', lw=1.5, label=f'±2σ')
            ax3.axhline(-2*sigma, color='red', linestyle=':', lw=1.5)
            
            ax3.set_xlabel(f'True {name}')
            ax3.set_ylabel('Residual')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        fig.suptitle('Comprehensive Regression Analysis Summary', fontsize=16, y=0.995)
        plt.tight_layout()
        return self._save_and_log(fig, 'test/comprehensive_summary')
    
    def print_statistics(self):
        """Print detailed statistics to console"""
        print("\n" + "="*80)
        print("DETAILED TEST STATISTICS")
        print("="*80)
        for i, name in enumerate(self.param_names):
            residuals = self.predictions[:, i] - self.labels[:, i]
            abs_errors = np.abs(residuals)
            
            print(f"\n{name}:")
            print(f"  MAE:     {abs_errors.mean():.6f}")
            print(f"  RMSE:    {np.sqrt((residuals**2).mean()):.6f}")
            print(f"  Max Err: {abs_errors.max():.6f}")
            print(f"  Median:  {np.median(abs_errors):.6f}")
            print(f"  Std:     {residuals.std():.6f}")
            print(f"  R²:      {1 - np.sum(residuals**2) / np.sum((self.labels[:, i] - self.labels[:, i].mean())**2):.6f}")
            
            # Percentiles
            print(f"  Error Percentiles:")
            for p in [50, 75, 90, 95, 99]:
                print(f"    {p}%: {np.percentile(abs_errors, p):.6f}")
        print("="*80 + "\n")
    
    def generate_all_plots(self, quick_mode=False):
        """
        Generate all evaluation plots.
        
        Args:
            quick_mode: If True, only generate essential plots (1, 2, 3, 10)
        """
        print("Generating evaluation plots...")
        
        # Essential plots (always generate)
        self.plot_predictions_vs_true()
        self.plot_residual_distributions()
        self.plot_metrics_comparison()
        
        if not quick_mode:
            # Advanced plots
            if self.n_outputs > 1:
                self.plot_residual_correlation()
            self.plot_error_vs_true()
            self.plot_qq()
            self.plot_comprehensive_summary()
        else:
            # Still generate comprehensive summary in quick mode
            self.plot_comprehensive_summary()
        
        # Always print statistics
        self.print_statistics()
        
        print(f"All plots saved to: {self.save_dir}")
#endregion --REGRESSION PLOTTER-----------------------------------------------------------
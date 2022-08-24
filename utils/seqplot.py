import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.Sequence import block, parula
import itertools
from pypulseq.calc_rf_center import calc_rf_center
import math
import mplcursors

def plot(
        seq,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True
    ) -> None:
        """
        Plot `Sequence`.

        Parameters
        ----------
        label : str, defualt=str()
            Plot label values for ADC events: in this example for LIN and REP labels; other valid labes are accepted as
            a comma-separated list.
        save : bool, default=False
            Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
            suffixes to the filename 'seq_plot'.
        show_blocks : bool, default=False
            Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        time_disp : str, default='s'
            Time display type, must be one of `s`, `ms` or `us`.
        grad_disp : str, default='s'
            Gradient display unit, must be one of `kHz/m` or `mT/m`.
        plot_now : bool, default=True
            If true, function immediately shows the plots, blocking the rest of the code until plots are exited.
            If false, plots are shown when plt.show() is called. Useful if plots are to be modified.
        plot_type : str, default='Gradient'
            Gradients display type, must be one of either 'Gradient' or 'Kspace'.
        """
        mpl.rcParams["lines.linewidth"] = 0.75  # Set default Matplotlib linewidth

        valid_time_units = ["s", "ms", "us"]
        valid_grad_units = ["kHz/m", "mT/m"]
        valid_labels = get_supported_labels()
        if (
            not all([isinstance(x, (int, float)) for x in time_range])
            or len(time_range) != 2
        ):
            raise ValueError("Invalid time range")
        if time_disp not in valid_time_units:
            raise ValueError("Unsupported time unit")

        if grad_disp not in valid_grad_units:
            raise ValueError("Unsupported gradient unit. Supported gradient units are: " + str(valid_grad_units))

        fig1 = plt.figure()
        sps = []
        sps.append(fig1.add_subplot(411))
        sps.append(fig1.add_subplot(412, sharex=sps[0]))
        sps.append(fig1.add_subplot(413, sharex=sps[0]))
        sps.append(fig1.add_subplot(414, sharex=sps[0]))


        t_factor_list = [1, 1e3, 1e6]
        t_factor = t_factor_list[valid_time_units.index(time_disp)]

        g_factor_list = [1e-3, 1e3/seq.system.gamma]
        g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

        t0 = 0
        label_defined = False
        label_idx_to_plot = []
        label_legend_to_plot = []
        label_store = dict()
        for i in range(len(valid_labels)):
            label_store[valid_labels[i]] = 0
            if valid_labels[i] in label.upper():
                label_idx_to_plot.append(i)
                label_legend_to_plot.append(valid_labels[i])

        if len(label_idx_to_plot) != 0:
            p = parula.main(len(label_idx_to_plot) + 1)
            label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))
            label_colors_to_plot = np.roll(label_colors_to_plot, -1, axis=0).tolist()

        # Block timings
        block_edges = np.cumsum([0, *seq.block_durations])
        block_edges_in_range = block_edges[
            (block_edges >= time_range[0]) * (block_edges <= time_range[1])
        ]
        if show_blocks:
            for sp in sps:
                sp.set_xticks(t_factor * block_edges_in_range)
                sp.set_xticklabels(rotation=90)

        for block_counter in range(len(seq.block_events)):
            block = seq.get_block(block_counter + 1)
            is_valid = time_range[0] <= t0 <= time_range[1]
            if is_valid:
                if getattr(block, "label", None) is not None:
                    for i in range(len(block.label)):
                        if block.label[i].type == "labelinc":
                            label_store[block.label[i].label] += block.label[i].value
                        else:
                            label_store[block.label[i].label] = block.label[i].value
                    label_defined = True

                if getattr(block, "adc", None) is not None:  # ADC
                    adc = block.adc
                    # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                    # is the present convention - the samples are shifted by 0.5 dwell
                    t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                    sps[1].vlines(x=t_factor * (t0 + t), ymin=-100, ymax=100, colors='b', linestyles='dashed')

                    # sps[1].plot(t_factor * (t0 + t), np.zeros(len(t)), "rx")
                    # sp13.plot(
                    #     t_factor * (t0 + t),
                    #     np.angle(
                    #         np.exp(1j * adc.phase_offset)
                    #         * np.exp(1j * 2 * np.pi * t * adc.freq_offset)
                    #     ),
                    #     "b.",
                    #     markersize=0.25,
                    # )

                    if label_defined and len(label_idx_to_plot) != 0:
                        cycler = mpl.cycler(color=label_colors_to_plot)
                        sps[1].set_prop_cycle(cycler)
                        label_colors_to_plot = np.roll(
                            label_colors_to_plot, -1, axis=0
                        ).tolist()
                        arr_label_store = list(label_store.values())
                        lbl_vals = np.take(arr_label_store, label_idx_to_plot)
                        t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                        _t = [t_factor * t] * len(lbl_vals)
                        # Plot each label individually to retrieve each corresponding Line2D object
                        p = itertools.chain.from_iterable(
                            [
                                sps[1].plot(__t, _lbl_vals, ".")
                                for __t, _lbl_vals in zip(_t, lbl_vals)
                            ]
                        )
                        if len(label_legend_to_plot) != 0:
                            sps[1].legend(p, label_legend_to_plot, loc="upper left")
                            label_legend_to_plot = []

                if getattr(block, "rf", None) is not None:  # RF
                    rf = block.rf
                    tc, ic = calc_rf_center(rf)
                    time = rf.t
                    signal = rf.signal
                    if np.abs(signal[0]) != 0:
                        signal = np.insert(signal, obj=0, values=0)
                        time = np.insert(time, obj=0, values=time[0])
                        ic += 1

                    if np.abs(signal[-1]) != 0:
                        signal = np.append(signal, 0)
                        time = np.append(time, time[-1])

                    sps[0].plot(t_factor * (t0 + time + rf.delay), np.abs(signal))
                    sps[0].plot(
                        t_factor * (t0 + time + rf.delay),
                        np.angle(
                            signal
                            * np.exp(1j * rf.phase_offset)
                            * np.exp(1j * 2 * math.pi * time * rf.freq_offset)
                        ),
                        t_factor * (t0 + tc + rf.delay),
                        np.angle(
                            signal[ic]
                            * np.exp(1j * rf.phase_offset)
                            * np.exp(1j * 2 * math.pi * time[ic] * rf.freq_offset)
                        ),
                        "xb",
                    )

                grad_channels = ["gx", "gy", "gz"]
                for x in range(len(grad_channels)):  # Gradients
                    if getattr(block, grad_channels[x], None) is not None:
                        grad = getattr(block, grad_channels[x])
                        if grad.type == "grad":
                            # We extend the shape by adding the first and the last points in an effort of making the
                            # display a bit less confusing...
                            time = grad.delay + [0, *grad.tt, grad.shape_dur]
                            waveform = g_factor * np.array(
                                (grad.first, *grad.waveform, grad.last)
                            )
                        else:
                            time = np.cumsum(
                                [
                                    0,
                                    grad.delay,
                                    grad.rise_time,
                                    grad.flat_time,
                                    grad.fall_time,
                                ]
                            )
                            waveform = g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                        sps[x+1].plot(t_factor * (t0 + time), waveform)
                        sps[x+1].set_ylim((np.min(waveform), np.max(waveform)))
            t0 += seq.block_durations[block_counter]

        grad_plot_labels = ["x", "y", "z"]
        # sp11.set_ylabel("ADC")
        sps[0].set_ylabel("RF mag (Hz)")
        # sp13.set_ylabel("RF/ADC phase (rad)")
        # sps[-1].set_xlabel(f"t ({time_disp})")
        for x in range(3):
            _label = grad_plot_labels[x]
            sps[x+1].set_ylabel(f"G{_label} ({grad_disp})")
        sps[-1].set_xlabel(f"t ({time_disp})")

        # Setting display limits
        disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
        [x.set_xlim(disp_range) for x in sps]

        # Grid on
        for sp in sps:
            sp.grid()

        fig1.tight_layout()

        mplcursors.cursor()
        if save:
            fig1.savefig("seq_plot1.jpg")
        
        if plot_now:
            plt.show()
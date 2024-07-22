from scipy import signal
import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import lmp_class as lmpc
from timeit import default_timer as timer

# file_path = sys.argv[1]
# file_name = sys.argv[2]
# delta_t = float(sys.argv[3]) * 1e-15
# outputf = sys.argv[4]


######## The constants will be used in this script ########
# c = 2.9979245899e10 # speed of light in vacuum in [cm/s], from Wikipedia.
# kB = 0.6950347      # Boltzman constant in [cm^-1/K], from Wikipedia.
# h_bar = 6.283185    # Reduced Planck constant in atomic unit, where h == 2*pi
# beta = 1.0/(kB * T) #

# c = 3966.2        # ang/sec longitudinal speed of sound in Cu:  c = sqrt(bulk modulus/density)
# c = 4720  #(m/s)  Fe

c = 4980.10
DT = 10e-15  # fs
NRSTEPS = 500
DIVISIONS = 8  # evenly spaced divisions

# -1)     simulation data: multiple spatial regions, uses DIVISIONS
# 0)      simulation data: single spatial region, uses XMIN and XMAX
# 1)      arbitrary function
# 2)      arbitrary function
# 3)

CHOICE = 0

path_to_data = "/"
lmp_filename = [path_to_data + "test.dump"]
outputfilename = path_to_data + "test.dat"


#############################################################################################
def get_indices(lmp, div, XMIN, XMAX):
    """
    return a list of lists.   indx contains a list of length nratoms, with each
    list containing a list of atom indices.
    """
    indx = []
    types = [[] for i in range(int(lmp.nrtypes))]
    for jj, val in enumerate(lmp.type):
        types[int(val - 1)].append(jj)
        if div == 0:
            # print('here', XMIN, XMAX, lmp.coordinate[jj,0])
            if lmp.coordinate[jj, 0] <= XMAX or lmp.coordinate[jj, 0] >= XMIN:
                indx.append(jj)
        else:
            if lmp.coordinate[jj, 0] <= XMAX and lmp.coordinate[jj, 0] >= XMIN:
                indx.append(jj)
    return types, indx


#############################################################################################
def choose_window(data, kind):  # window_function_name):
    # kind = input(window_function_name)
    standard = 10

    if kind == "Gaussian":
        sigma = 2 * math.sqrt(2 * math.log(2))
        std = float(standard)
        window_function = signal.gaussian(len(data), std / sigma, sym=False)

    elif kind == "Blackman-Harris":
        window_function = signal.blackmanharris(len(data), sym=False)

    elif kind == "Hamming":
        window_function = signal.hamming(len(data), sym=False)

    elif kind == "Hann":
        window_function = signal.hann(len(data), sym=False)

    return window_function


#############################################################################################
def zero_padding(sample_data):
    """A series of Zeros will be padded to the end of the dipole moment
    array (before FFT performed), in order to obtain a array with the
    length which is the "next power of two" of numbers.

    #### Next power of two is calculated as: 2**math.ceil(math.log(x,2))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    """
    return int(2 ** math.ceil(math.log(len(sample_data), 2)))


#############################################################################################
def calc_ACF(array_1D):
    # Normalization
    yunbiased = array_1D - np.mean(array_1D, axis=0)
    ynorm = np.sum(np.power(yunbiased, 2), axis=0)

    # autocor = signal.fftconvolve(array_1D, array_1D[::-1], mode='full')[len(array_1D)-1:] / ynorm
    # autocor = signal.fftconvolve(array_1D, array_1D[::-1], mode='full') #/ ynorm

    autocor = signal.correlate(array_1D, array_1D, mode="full", method="fft")

    return autocor


#############################################################################################
def calc_FFT(array_1D):  # , window):
    """
    This function is for calculating the "intensity" of the ACF at each frequency
    by using the discrete fast Fourier transform.
    """
    ####
    #### http://stackoverflow.com/questions/20165193/fft-normalization
    ####
    window = choose_window(array_1D, "Gaussian")
    WE = sum(window) / len(array_1D)
    wf = window / WE
    # convolve the blackman-harris window function.

    # sig = array_1D * wf
    sig = array_1D

    # A series of number of zeros will be padded to the end of the \
    # VACF array before FFT.
    N = zero_padding(sig)

    yfft = np.fft.fft(sig, N, axis=0) / len(sig)
    #    yfft = np.fft.fft(data, n=int(N_fft), axis=0)/len(data) # no window func.
    #    print("shape of yfft {:}".format(np.shape(yfft)))
    return np.absolute(yfft)  # np.square(np.absolute(yfft))


#############################################################################################
def save_results(fout, wavenumber, intensity):
    with open(fout, "w") as fw:
        title = ("Wavenumber", "IR Intensity", "cm^-1", "a.u.")
        np.savetxt(
            fout,
            np.c_[wavenumber, intensity],
            fmt="%10.5f %15.5e",
            header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
            comments="",
        )


################################################################################################
def plot_stuff(time, time2, sig, acf, freq, freq2, intensity, intensity2):

    #    plt.figure(1, figsize=[6,8],clear=True)
    #    plt.subplot(4,1,1)
    #    plt.plot(time, sig, color="k", marker='o', markersize=2.0)
    #    plt.xlabel("Time", fontsize=15)
    #    plt.ylabel("Signal", fontsize=15)

    #    plt.subplot(4,1,2)
    #    plt.plot(time2, acf, color="k") #, marker='o', markersize=2.0)
    #    plt.xlabel("Time", fontsize=15)
    #    plt.ylabel("ACF(signal)", fontsize=15)

    #    plt.subplot(4,1,3)
    #    plt.plot(freq, intensity, color="k") #, marker='o', markersize=2.0)
    #    plt.xlabel("Freq (1/time)", fontsize=15)
    #    plt.ylabel("|fft(signal)|", fontsize=15)

    #    plt.subplot(4,1,4)
    #    plt.plot(freq2, intensity2, color="k", marker='o', markersize=1.0)
    #    plt.xlabel("Freq (1/time)", fontsize=15)
    #    plt.ylabel("|fft(ACF)|", fontsize=15)

    #    plt.subplots_adjust(hspace = 0.5)
    #    plt.show()

    plt.figure(1, figsize=[6, 8], clear=True)
    plt.subplot(3, 1, 1)
    plt.plot(time, sig, color="k", marker="o", markersize=2.0)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Signal", fontsize=15)

    plt.subplot(3, 1, 2)
    plt.plot(time2, acf, color="k")  # , marker='o', markersize=2.0)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("ACF(signal)", fontsize=15)

    plt.subplot(3, 1, 3)
    plt.plot(freq2, intensity2, color="k", marker="o", markersize=4.0)
    plt.xlabel("Freq (1/time)", fontsize=15)
    plt.ylabel("|fft(ACF)|", fontsize=15)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return


#############################################################################################
def main():

    if CHOICE == 1:
        L = 1500  # user will know this value - total number of timesteps
        T = 1 / 500.0  # dt; user will know this
        Fs = 1 / T  # 1/dt

        time = np.arange(0, L - 1) * T  # sim data

        print(np.min(time), np.max(time))

        print("period: ", T, "frequency: ", Fs, "length: ", L)
        # sig = 0.5*np.sin(2*np.pi*50*time) + 2*np.sin(2*np.pi*520*time)   # input from simulation
        # sig = 0.5*np.sin(2*np.pi*50*time) + 1*np.sin(2*np.pi*169*time) + 2*np.sin(2*np.pi*520*time)
        sig = 0.7 * np.sin(2 * np.pi * 50 * time) + 1 * np.sin(2 * np.pi * 120 * time)
        # sig += np.random.rand(L)  # random noise?

        yfft = calc_FFT(sig)
        intensity = 2.0 * yfft[0 : int(L / 2)]
        freq = np.fft.fftfreq(n=len(yfft), d=T)[0 : int(len(intensity))]

        plt.subplot(2, 1, 1)
        plt.plot(time, sig, color="red")
        plt.xlabel("Time", fontsize=15)
        plt.ylabel("Signal", fontsize=15)
        plt.subplot(2, 1, 2)
        plt.plot(freq, intensity, color="red", marker="o", markersize=2.0)
        plt.xlabel("Freq (1/time)", fontsize=15)
        plt.ylabel("|fft(signal)|", fontsize=15)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return

    elif CHOICE == 2:

        N = 2000
        time = np.linspace(0, 1, N)  # this could be input from simulations
        time2 = np.concatenate([-time[::-1][:-1], time])
        L = len(time)
        L2 = len(time2)
        T = time[1] - time[0]  # "sampling period" --> dt of the simulation
        fs = 1 / T
        # "sampling frequency"

        print("period: ", T, "frequency: ", fs, "length: ", L)
        #      sig = 0.5*np.sin(2*np.pi*13*time) + 2*np.sin(2*np.pi*34*time)   # input from simulation
        sig = (
            0.5 * np.sin(2 * np.pi * 50 * time)
            + 1 * np.sin(2 * np.pi * 169 * time)
            + 2 * np.sin(2 * np.pi * 520 * time)
        )
        #        sig += np.random.rand(L)  # random noise?

        yfft = calc_FFT(sig)
        intensity = 2.0 * yfft[0 : int(L / 2) + 1]
        freq = np.fft.fftfreq(n=len(yfft), d=T)[0 : int(len(intensity))]

        acf = calc_ACF(sig)
        yfft2 = calc_FFT(acf)
        intensity2 = 2.0 * yfft2[0 : int(L2 / 2) + 1]
        freq2 = np.fft.fftfreq(n=len(yfft2), d=T)[0 : int(len(intensity2))]

        plot_stuff(time, time2, sig, acf, freq, freq2, intensity, intensity2)

        return

    elif CHOICE == 0:

        # ------------ initialize things
        XMIN = -100000
        XMAX = 100000

        start = timer()
        lmp = lmpc.obj(filename=lmp_filename)
        vel = np.zeros(shape=(NRSTEPS, lmp.nratoms, 3), dtype=float)
        norm_velocity = np.zeros(shape=(NRSTEPS, lmp.nratoms), dtype=float)
        types, indx = get_indices(
            lmp, 1, XMIN, XMAX
        )  # indx contains atom_ids that are within xrange
        nratoms = len(indx)
        print("Using ", nratoms, " out of ", lmp.nratoms)

        # ------------ get all velocity data
        for jj in range(NRSTEPS):
            if lmp.errorflag != 0:
                break
            vel[jj, :, :] = lmp.velocity

            # norm_velocity[jj,:] = lmp.velocity[:,0]

            norm_velocity[jj, :] = np.linalg.norm(vel[jj, :, :], axis=-1)
            lmp.next_step()

        # ---------- calculate VACF and fft
        intensity = np.zeros(shape=(int(NRSTEPS / 2) + 1, 4), dtype=float)
        ave_vacf = np.zeros(shape=(int(NRSTEPS) * 2 - 1, 4), dtype=float)

        cnt = 0
        for i in range(lmp.nratoms):
            if i not in indx:
                continue
            cnt += 1

            ACF = calc_ACF(vel[:, i, 0])
            ave_vacf[:, 0] += ACF
            yfft = calc_FFT(ACF)
            intensity[:, 0] += yfft[0 : int(NRSTEPS / 2) + 1]

            ACF = calc_ACF(vel[:, i, 1])
            ave_vacf[:, 1] += ACF
            yfft = calc_FFT(ACF)
            intensity[:, 1] += yfft[0 : int(NRSTEPS / 2) + 1]

            ACF = calc_ACF(vel[:, i, 2])
            ave_vacf[:, 2] += ACF
            yfft = calc_FFT(ACF)
            intensity[:, 2] += yfft[0 : int(NRSTEPS / 2) + 1]

            ACF = calc_ACF(norm_velocity[:, i])
            ave_vacf[:, 3] += ACF
            yfft = calc_FFT(ACF)
            intensity[:, 3] += yfft[0 : int(NRSTEPS / 2) + 1]

        intensity /= cnt
        ave_vacf /= cnt

        freq = np.fft.fftfreq(n=len(yfft), d=DT)[0 : int(intensity.shape[0])]
        #        wavenumber = c/freq #/2/np.pi
        df = freq[1] - freq[0]

        for i in range(intensity.shape[1]):
            intensity[:, i] /= np.sum(intensity[:, i] * df)
            intensity[:, i] *= 3 * cnt
            print(i, np.sum(intensity[:, i] * df), np.sum(intensity[:, i]), df)

        # plt.figure(1)
        # plt.plot(ave_vacf[:,0], color="black", marker='o', markersize=2.0)
        # plt.xlabel('Frequency (HZ)',fontsize=20)
        # plt.ylabel('PDOS ',fontsize=20)
        # plt.show()

        fidout = open(outputfilename, "w")
        fidout.write("freq   pdos_x   pdos_y  pdos_z   pdos^2 \n")
        for ll in range(freq.shape[0]):
            fidout.write(
                " %15.10e  %15.10e  %15.10e  %15.10e  %15.10e\n"
                % (
                    freq[ll],
                    intensity[ll, 0],
                    intensity[ll, 1],
                    intensity[ll, 2],
                    intensity[ll, 3],
                )
            )
        fidout.close()

        # --------- clean up after yourself
        lmp.file_close()
        end = timer()
        print("RUN DURATION: ", (end - start), " [sec]")
        print(
            "Memory Usage: %f [Gb]"
            % (NRSTEPS * lmp.nratoms * 3 * 32 / 8 / 1024 / 1024 / 1024.0)
        )

    elif CHOICE == -1:

        # ------------ initialize things
        start = timer()
        lmp = lmpc.obj(filename=lmp_filename)
        vel = np.zeros(shape=(NRSTEPS, lmp.nratoms, 3), dtype=float)
        norm_velocity = np.zeros(shape=(NRSTEPS, lmp.nratoms), dtype=float)
        master_intensity = np.zeros(
            shape=(int(NRSTEPS / 2) + 1, 4, DIVISIONS), dtype=float
        )

        # ------------ get all velocity data
        for jj in range(NRSTEPS):
            if lmp.errorflag != 0:
                break
            vel[jj, :, :] = lmp.velocity

            # norm_velocity[jj,:] = lmp.velocity[:,0]

            norm_velocity[jj, :] = np.linalg.norm(vel[jj, :, :], axis=-1)
            lmp.next_step()

        # ---------- calculate VACF and fft
        dx = (
            (lmp.box[1] - lmp.box[0]) / DIVISIONS / 2
        )  # assumes bar length is in x-direction
        divs = np.arange(lmp.box[0] + dx, lmp.box[1], 2 * dx)

        #        print(divs)

        master_intensity = np.zeros(
            shape=(int(NRSTEPS / 2) + 1, 4, DIVISIONS), dtype=float
        )

        for jj in range(DIVISIONS):

            print("WORKING ON REGION %d" % jj)

            if jj == 0:
                XMIN = divs[-1]
                XMAX = divs[0]
            else:
                XMIN = divs[jj - 1]
                XMAX = divs[jj]

            # print(XMIN, XMAX)

            types, indx = get_indices(
                lmp, jj, XMIN, XMAX
            )  # indx contains atom_ids that are within xrange
            nratoms = len(indx)
            print("Using ", nratoms, " out of ", lmp.nratoms, " atoms")

            intensity = np.zeros(shape=(int(NRSTEPS / 2) + 1, 4), dtype=float)
            ave_vacf = np.zeros(shape=(int(NRSTEPS) * 2 - 1, 4), dtype=float)

            cnt = 0
            for i in range(lmp.nratoms):
                if i not in indx:
                    continue
                cnt += 1

                ACF = calc_ACF(vel[:, i, 0])
                ave_vacf[:, 0] += ACF
                yfft = calc_FFT(ACF)
                intensity[:, 0] += yfft[0 : int(NRSTEPS / 2) + 1]

                ACF = calc_ACF(vel[:, i, 1])
                ave_vacf[:, 1] += ACF
                yfft = calc_FFT(ACF)
                intensity[:, 1] += yfft[0 : int(NRSTEPS / 2) + 1]

                ACF = calc_ACF(vel[:, i, 2])
                ave_vacf[:, 2] += ACF
                yfft = calc_FFT(ACF)
                intensity[:, 2] += yfft[0 : int(NRSTEPS / 2) + 1]

                ACF = calc_ACF(norm_velocity[:, i])
                ave_vacf[:, 3] += ACF
                yfft = calc_FFT(ACF)
                intensity[:, 3] += yfft[0 : int(NRSTEPS / 2) + 1]

            intensity /= cnt
            ave_vacf /= cnt

            freq = np.fft.fftfreq(n=len(yfft), d=DT)[0 : int(intensity.shape[0])]
            #            wavenumber = c/freq #/2/np.pi
            df = freq[1] - freq[0]

            for i in range(intensity.shape[1]):
                intensity[:, i] /= np.sum(intensity[:, i] * df)
                intensity[:, i] *= 3 * cnt
                print(i, np.sum(intensity[:, i] * df), np.sum(intensity[:, i]), df)

            master_intensity[:, :, jj] = intensity

        # ---------- write master
        fidout = open(outputfilename, "w")
        fidout.write("freq")
        for ii in range(DIVISIONS):
            fidout.write(
                "   %dpdos_x   %dpdos_y  %dpdos_z   %dpdos^2 " % (ii, ii, ii, ii)
            )
        fidout.write("\n")

        for ll in range(freq.shape[0]):
            fidout.write(" %15.6e " % freq[ll])
            for ii in range(DIVISIONS):
                fidout.write(
                    " %15.6e  %15.6e  %15.6e  %15.6e"
                    % (
                        master_intensity[ll, 0, ii],
                        master_intensity[ll, 1, ii],
                        master_intensity[ll, 2, ii],
                        master_intensity[ll, 3, ii],
                    )
                )
            fidout.write("\n")
        fidout.close()

        # --------- clean up after yourself
        lmp.file_close()
        end = timer()
        print("RUN DURATION: ", (end - start), " [sec]")
        print(
            "Memory Usage: %f [Gb]"
            % (NRSTEPS * lmp.nratoms * 3 * 32 / 8 / 1024 / 1024 / 1024.0)
        )


#############################################################################################
######## The main program ########
if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import math
import lmp_class as lmpc
from timeit import default_timer as timer

NUM_BINS = 600  # a binsize of 0.1 angstom seems appropriate
MIN_VAL = 0.0
MAX_VAL = 30.0  # 100.0


path_to_data = "/"
lmp_filename = [path_to_data + ".dump"]
write_steps = [50, 500]
outputfilename = "rdf_test.dat"


# -------------------------------------------------------------------------------
def get_indices(lmp):
    """
    return a list of lists.   indx contains a list of length nrtypes, with each
    list containing a list of atom indices.
    """
    indx = [[] for i in range(int(lmp.nrtypes))]
    for jj, val in enumerate(lmp.type):
        indx[int(val - 1)].append(jj)

    return indx


# -----------------------------------------------------------------------------
def distance_histogram(lmp, type_ii, type_jj, xsize, ysize, zsize):

    dist = np.zeros(shape=(NUM_BINS), dtype=float)

    for ids, ii in enumerate(type_ii):

        temp_x = lmp.coordinate[type_jj, 0] - lmp.coordinate[ii, 0]
        temp_y = lmp.coordinate[type_jj, 1] - lmp.coordinate[ii, 1]
        temp_z = lmp.coordinate[type_jj, 2] - lmp.coordinate[ii, 2]

        # ------- correct for PBC
        temp_x[abs(temp_x) > xsize / 2.0] -= (
            np.sign(temp_x[abs(temp_x) > xsize / 2.0]) * xsize
        )
        temp_y[abs(temp_y) > ysize / 2.0] -= (
            np.sign(temp_y[abs(temp_y) > ysize / 2.0]) * ysize
        )
        temp_z[abs(temp_z) > zsize / 2.0] -= (
            np.sign(temp_z[abs(temp_z) > zsize / 2.0]) * zsize
        )

        d = np.sqrt((temp_x) ** 2 + (temp_y) ** 2 + (temp_z) ** 2)
        d = d[d != 0.0]

        temp, edges = np.histogram(d, bins=NUM_BINS, range=(MIN_VAL, MAX_VAL))

        # print(dist.shape)
        # print(temp.shape)

        dist += temp

    return dist, edges


# ------------------------------------------------------------------------------
def main_script():

    start = timer()

    # -----  lammps object
    if os.path.isfile(lmp_filename[0]) == False:
        print("ERROR:  file does not exist")
        print(lmp_filename[0])
        return

    lmp = lmpc.obj(filename=lmp_filename)
    types = get_indices(lmp)
    number_types = [len(x) for x in types]
    N = math.comb(len(types) + 1, 2)

    # ------- this is the array to build/write
    rdf = np.zeros(shape=(NUM_BINS, N), dtype=float)

    cnt = 0
    nrsteps = 0
    while True:  # timesteps
        # for zz in range(50):

        if lmp.errorflag != 0:
            break

        cnt += 1
        nrsteps += 1
        xsize = lmp.box[1] - lmp.box[0]
        ysize = lmp.box[3] - lmp.box[2]
        zsize = lmp.box[5] - lmp.box[4]
        vol = xsize * ysize * zsize

        if cnt == 1:
            valid_to = min([xsize, ysize, zsize])
            fidout = open(outputfilename, "w")
            fidout.write(" rdf is valid to r= %f\n" % (valid_to / 2.0))
            fidout.write(" r  ")
            for ii in range(lmp.nrtypes):  # sit on this type
                for jj in range(ii, lmp.nrtypes):
                    fidout.write("  g(r)_%d-%d " % (ii + 1, jj + 1))
            fidout.write("\n")

        nn = 0
        for ii in range(lmp.nrtypes):  # sit on this type
            for jj in range(ii, lmp.nrtypes):  # calc to this type
                hist, edges = distance_histogram(
                    lmp, types[ii], types[jj], xsize, ysize, zsize
                )
                rdf[:, nn] += hist * vol / len(types[jj]) / len(types[ii])
                nn += 1

        if cnt in write_steps:
            # -------------- normalization
            ee = (edges[:-1] + edges[1:]) / 2.0
            for kk in range(N):
                for ll in range(len(edges) - 1):
                    rdf[ll, kk] /= (
                        4 / 3.0 * math.pi * (edges[ll + 1] ** 3 - edges[ll] ** 3)
                    )
                    rdf[ll, kk] /= nrsteps

            # write output
            fidout.write("STEP: %d \n" % lmp.timestep)
            for kk in range(NUM_BINS):
                fidout.write("%15.6f  " % ee[kk])
                for ll in range(N):
                    fidout.write("%15.6f " % (rdf[kk, ll]))
                fidout.write("\n")

            # -------- reset
            rdf = np.zeros(
                shape=(NUM_BINS, N), dtype=float
            )  # it can support upto 10, but I think 6 is max
            nrsteps = 0

        # nest step
        lmp.next_step()

    # clean up
    fidout.close()

    end = timer()
    print("RUN DURATION: ", (end - start), " [sec]")

    return


# -------------------------------------------------------------------------------
if __name__ == "__main__":

    main_script()

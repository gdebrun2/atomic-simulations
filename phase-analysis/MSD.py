

'''

# MSD.py 
# Calculate the mean squared displacement using multiple traces 
# 3/19/21

#  Author:
#        Mark A. Wilson
#        Sandia National Laboratories
#        Computational Materials and Data Science (Org. 1864)
#        marwils@sandia.gov
  
'''

import os
import sys
import numpy as np
import math
import datetime
import lmp_class as lmpc
from timeit import default_timer as timer


A_BIG_NUMBER = 20000      # max number of timesteps....  sets array sizes
MAX_NR_TYPES = 7          # max number ofatom types to start with.  This accounts for possible changes in types
#STEPS_TO_START_TRACE = [0, 10, 50, 100, 200, 300, 400, 500, 750, 1000]        # must include 0
STEPS_TO_START_TRACE = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70] 


#concentration = sys.argv[1]
#temperature = sys.argv[2]
#directory = sys.argv[3]
#version = sys.argv[4]

#path_to_data = '/nscratch/marwils/HMat/msd/' + temperature +'K/' + directory + '/'
#lmp_filename = [path_to_data + 'config_relax' + version + '_eq' + temperature + 'K_1atm_' + concentration + '_a.dump']
#outputfilename = path_to_data + 'config_relax' + version + '_eq' + temperature + 'K_1atm_' + concentration + '.msd'

#lmp_filename = [path_to_data + 'config_relax1_eq' + temperature + 'K_1atm_' + concentration + '_a.dump', 
#		path_to_data + 'config_relax1_eq' + temperature + 'K_1atm_' + concentration + '_b.dump',
#		path_to_data + 'config_relax1_eq' + temperature + 'K_1atm_' + concentration + '_c.dump',
#		path_to_data + 'config_relax1_eq' + temperature + 'K_1atm_' + concentration + '_d.dump'] 
#outputfilename = path_to_data + 'config_relax1_eq' + temperature + 'K_1atm_' + concentration + '.msd'

write_all_traces = 0

#path_to_data = '/Users/marwils/lammps-29Oct20/examples/melt/'
#lmp_filename = [path_to_data + 'msd_test_all.dump']
#outputfilename = path_to_data + 'msd_test_all.msd_me'

path_to_data = '/Users/marwils/research/TimeGEMM/coarse/general/production/'
lmp_filename = [path_to_data + 'CG_N101_XL11_C100_rep1.dump']
outputfilename = path_to_data + 'CG_N101_XL11_C100_rep1.msd'

temperature = sys.argv[1]
path_to_data = '/Users/marwils/research/gas-surface/insert_particle/molecule/'
lmp_filename = [path_to_data + 'production_noq_' + str(temperature) + 'K_a.dump']
outputfilename = path_to_data + 'production_noq_' + str(temperature) + 'K_a.msd'


#-------------------------------------------------------------------------------
def get_indices(lmp):
     '''  
         return a list of lists.   indx contains a list of length nratoms, with each 
         list containing a list of atom indices. 
     '''
     indx = [[] for i in range(int(lmp.nrtypes))]
     for jj, val in enumerate(lmp.type):
         indx[int(val[0]-1)].append(jj)

     return indx  # starting at 0

#-------------------------------------------------------------------------------
def calc_displacement_update_flags(lmp, ref):
     '''
         Determine the displacement 
     '''
     flags = np.zeros(shape=(lmp.nratoms,3), dtype=int)

     disp = np.zeros(shape=(lmp.nratoms,1), dtype=float)   # assumes that lmp.nratoms doesnt change
     disp_x = np.zeros(shape=(lmp.nratoms,1), dtype=float)
     disp_y = np.zeros(shape=(lmp.nratoms,1), dtype=float)
     disp_z = np.zeros(shape=(lmp.nratoms,1), dtype=float)

     xsize = (lmp.box[1] - lmp.box[0])
     ysize = (lmp.box[3] - lmp.box[2])
     zsize = (lmp.box[5] - lmp.box[4])

     for jj in range(lmp.nratoms):

         temp_x = lmp.coordinate[jj,0]-ref.coordinate[jj,0] 
         temp_y = lmp.coordinate[jj,1]-ref.coordinate[jj,1]
         temp_z = lmp.coordinate[jj,2]-ref.coordinate[jj,2]

         # ------- correct for PBC
         if abs(temp_x) > xsize/2.0: 
             flags[jj,0] -= 1*np.sign(temp_x)
             temp_x -= np.sign(temp_x)*xsize
         if abs(temp_y) > ysize/2.0: 
             flags[jj,1] -= 1*np.sign(temp_y)
             temp_y -= np.sign(temp_y)*ysize
         if abs(temp_z) > zsize/2.0: 
             flags[jj,2] -= 1*np.sign(temp_z)
             temp_z -= np.sign(temp_z)*zsize

         disp[jj] = math.sqrt( (temp_x)**2 + (temp_y)**2 + (temp_z)**2)
         disp_x[jj] = temp_x
         disp_y[jj] = temp_y
         disp_z[jj] = temp_z

     return disp, disp_x, disp_y, disp_z, flags



#-------------------------------------------------------------------------------
def calc_displacement_use_flags(lmp, ref, flags):
     '''
         calculate displacement back to t0 using imageflages 
     '''
     disp = np.zeros(shape=(lmp.nratoms,1), dtype=float)   # assumes that lmp.nratoms doesnt change
     disp_x = np.zeros(shape=(lmp.nratoms,1), dtype=float)
     disp_y = np.zeros(shape=(lmp.nratoms,1), dtype=float)
     disp_z = np.zeros(shape=(lmp.nratoms,1), dtype=float)
     dx = lmp.box[1] - ref.box[1]

     xsize = (lmp.box[1] - lmp.box[0])
     ysize = (lmp.box[3] - lmp.box[2])
     zsize = (lmp.box[5] - lmp.box[4])

     for jj in range(lmp.nratoms):

         temp_x = lmp.coordinate[jj,0]-ref.coordinate[jj,0] + flags[jj,0] * xsize 
         temp_y = lmp.coordinate[jj,1]-ref.coordinate[jj,1] + flags[jj,1] * ysize
         temp_z = lmp.coordinate[jj,2]-ref.coordinate[jj,2] + flags[jj,2] * zsize

         disp[jj] = (temp_x)**2 + (temp_y)**2 + (temp_z)**2
         disp_x[jj] = temp_x
         disp_y[jj] = temp_y
         disp_z[jj] = temp_z

     return disp, disp_x, disp_y, disp_z

#-------------------------------------------------------------------------------
def main_script():


    start = timer()    

  # -----  lammps object 
    if os.path.isfile(lmp_filename[0]) == False: 
        print('ERROR: input dumpfile does not exist', lmp_filename[0])
        return

    lmp = lmpc.obj(filename=lmp_filename)
    lmp_ref = lmp.copy()

  # ------ indices
    indx = get_indices(lmp)
#    ave_msd = np.zeros(shape=(A_BIG_NUMBER,int(lmp.nrtypes)*4+1), dtype=float)
#    tracecount = np.zeros(shape=(A_BIG_NUMBER,int(lmp.nrtypes)), dtype=int)

    ave_msd = np.zeros(shape=(A_BIG_NUMBER,MAX_NR_TYPES*4), dtype=float)
    tracecount = np.zeros(shape=(A_BIG_NUMBER,MAX_NR_TYPES), dtype=int)
    all_msd = np.zeros(shape=(A_BIG_NUMBER,4), dtype=float)
    all_tracecount = np.zeros(shape=(A_BIG_NUMBER), dtype=int)
    time = np.zeros(shape=(A_BIG_NUMBER), dtype=int)

    imgflgs = []
    refs = []
    step = 0

  # -------- iterate through all timesteps in the file

#    for mm in range(30):
    while(1):
         if lmp.errorflag != 0: break 

         indx = []
         indx = get_indices(lmp)  # this is to account for possible changing types

        # ------------- reference snapshots
         if step in STEPS_TO_START_TRACE:
             r = lmp.copy()
             refs.append(r)
             ifs = np.zeros(shape=(lmp.nratoms,3), dtype=int)
             imgflgs.append(ifs)
#             print('new starting point')

         if lmp.image_flag.shape[0] == 0:
             dummy, dummy_x, dummy_y, dummy_z, imageflags = calc_displacement_update_flags(lmp, lmp_ref)
         else:
             print('found image_flags')
             imageflags = lmp.image_flag - lmp_ref.image_flag

         for kk, lmp_orig in enumerate(refs):  # for all the start time traces...

            # ----- update image_flags
             if step == STEPS_TO_START_TRACE[kk]: 
                 imgflgs[kk] = np.zeros(shape=(lmp.nratoms,3), dtype=int)
             else:    
                 imgflgs[kk] += imageflags

             disp_sqr, disp_x, disp_y, disp_z   = calc_displacement_use_flags(lmp, lmp_orig, imgflgs[kk])
             dstep = step - STEPS_TO_START_TRACE[kk]   # updated step [int]

             if kk == 0:
                 time[step] = lmp.timestep                     
             for ii in range(int(lmp.nrtypes)):
#                 print('here4 ---------- ',dstep, ii)
                 if indx[ii]:
                     tracecount[dstep,ii] += 1
                     ave_msd[dstep,ii*4] += np.mean(disp_x[indx[ii]]**2)
                     ave_msd[dstep,ii*4+1] += np.mean(disp_y[indx[ii]]**2)
                     ave_msd[dstep,ii*4+2] += np.mean(disp_z[indx[ii]]**2)
                     ave_msd[dstep,ii*4+3] += np.mean(disp_sqr[indx[ii]])
                 else:
                     ave_msd[dstep,ii*4] += 0.0
                     ave_msd[dstep,ii*4+1] += 0.0
                     ave_msd[dstep,ii*4+2] += 0.0
                     ave_msd[dstep,ii*4+3] += 0.0

         # ---------the entire trace
             all_tracecount[dstep] += 1
             all_msd[dstep,0] += np.mean(disp_x**2)
             all_msd[dstep,1] += np.mean(disp_y**2)
             all_msd[dstep,2] += np.mean(disp_z**2)
             all_msd[dstep,3] += np.mean(disp_sqr)

      # -------- moving to next step
         lmp_ref = lmp.copy() 
         lmp.next_step()
         step += 1

   # ---------- write output; normalize by the number of traces
#    print(all_msd.shape())

#    print(all_tracecount.shape())
    for ll in range(A_BIG_NUMBER):
        if ave_msd[ll,0] != 0 or ll == 0:
            for kk in range(all_msd.shape[1]): 
                if all_tracecount[ll] > 0:
                    all_msd[ll,kk] /= all_tracecount[ll]
                else:
                    all_msd[ll,kk] = 0.0
            for kk in range(MAX_NR_TYPES):
                if tracecount[ll,kk] > 0:
                    pass
                    #ave_msd[ll,kk*4] /= tracecount[ll,kk]
                    #ave_msd[ll,kk*4+1] /= tracecount[ll,kk]
                    #ave_msd[ll,kk*4+2] /= tracecount[ll,kk]
                    #ave_msd[ll,kk*4+3] /= tracecount[ll,kk]
                else:
                    ave_msd[ll,kk*4] = 0.0
                    ave_msd[ll,kk*4+1] = 0.0
                    ave_msd[ll,kk*4+2] = 0.0 
                    ave_msd[ll,kk*4+3] = 0.0

   # -------- write the output
    fidout = open(outputfilename, 'w')
    fidout.write('STEP  ')
    fidout.write(" x^2_all   y^2_all   z^2_all   msd^2_all  ")
    for ii in range(MAX_NR_TYPES):
        fidout.write(" x^2_%d   y^2_%d   z^2_%d   msd^2_%d  " %(ii+1, ii+1, ii+1, ii+1))
    fidout.write('\n')
    for kk in range(A_BIG_NUMBER):    
         if ave_msd[kk,0] != 0 or kk == 0:
             fidout.write(' %15d ' %(time[kk]))
             for ll in range(all_msd.shape[1]):
                 fidout.write(' %15.6f ' %( all_msd[kk,ll] ))
             for ll in range(ave_msd.shape[1]):
                 fidout.write(' %15.6f ' %( ave_msd[kk,ll] ))
             fidout.write('\n')


   # ----------- clean up after yourself
    fidout.close()
#    fid2.close()
    lmp.file_close()

    end = timer()
    print ("RUN DURATION: ", (end - start), " [sec]")


    return

#-------------------------------------------------------------------------------
if __name__ == '__main__':


    main_script()















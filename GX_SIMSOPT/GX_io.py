import numpy as np
from netCDF4 import Dataset

import subprocess
import os

class GX_Runner():

    # This class handles GX input files, and also execution

    def __init__(self,template):
        
        self.read_input(template)


    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('#') > -1:
                end = line.find('#')
                line = line[:end]

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key   = key.strip()
            value = value.strip()
            
            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        self.inputs = obj


    def write(self, fout='temp.in'):

        # do not overwrite
        if (os.path.exists(fout)):
            print( '  input exists, skipping write', fout )
            return

        with open(fout,'w') as f:
        
            for item in self.inputs.items():
                 
                if ( type(item[1]) is not dict ):
                    print('  %s = %s ' % item, file=f)  
                    continue
    
                header, nest = item
                print('\n[%s]' % header, file=f)
    
                longest_key =  max( nest.keys(), key=len) 
                N_space = len(longest_key) 
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)

        print('  wrote input:', fout)

    def execute(self):

        # assume Trinity is in a salloc environment with GPUs
        # write input file, write batch file, execute srun
        pass

    def pretty_print(self, entry=''):
    # dumps out current input data, in GX input format
    #    if entry, where entry is one of the GX input headers
    #       only print the inputs nested under entry

        for item in self.inputs.items():
        
            # a catch for the debug input, which has no header
            if ( type(item[1]) is not dict ):
                if (entry == ''):
                    print('  %s = %s ' % item)
                    continue
     
            header, nest = item

            # special case
            if (entry != ''):
                header = entry
                nest   = self.inputs[entry]

            print('\n[%s]' % header)
     
            longest_key =  max( nest.keys(), key=len) 
            N_space = len(longest_key) 
            for pair in nest.items():
                s = '  {:%i}  =  {}' % N_space
                print(s.format(*pair))

            # special case
            if (entry != ''):
                break


# should this be a class? Yes, this is now outdated 8/20
def read_GX_output(fname):
    
    try:
        f = Dataset(fname, mode='r')
    except: 
        print('  read_GX_output: could not read', fname)


    t = f.variables['time'][:]
    q = f.groups['Fluxes'].variables['qflux'][:,0]

    # check for NANs
    if ( np.isnan(q).any() ):
         print('  nans found in', fname)
         q = np.nan_to_num(q)

    # median of a sliding median
    N = len(q)
    med = np.median( [ np.median( q[::-1][:k] ) for k in np.arange(1,N)] )

    #print('  read GX output: qflux = ', med)
    return med # this is the qflux

class GX_Output():

    def __init__(self,fname):

        try:
            f = Dataset(fname, mode='r')
            #f = nc.netcdf_file(fname, 'r') 
        except: 
            print('  read_GX_output: could not read', fname)
    
    
        qflux = f.groups['Fluxes'].variables['qflux'][:,0]
    
        # check for NANs
        if ( np.isnan(qflux).any() ):
             print('  nans found in', fname)
             qflux = np.nan_to_num(qflux)

        self.qflux = qflux
        self.time  = f.variables['time'][:]

        self.tprim  = f.groups['Inputs']['Species']['T0_prime'][:]
        self.fprim  = f.groups['Inputs']['Species']['n0_prime'][:]

        self.fname = fname
        self.data = f

    def median_estimator(self):

        N = len(self.qflux)
        med = np.median( [ np.median( self.qflux[::-1][:k] ) for k in np.arange(1,N)] )

        self.q_median = med
        return med

    def exponential_window_estimator(self, tau=100):

        # load data
        time  = self.time
        qflux = self.qflux

        # initial state
        t0       = time[0]
        qavg     = qflux[0]
        var_qavg = 0
        
        Q_avg = []
        Var_Q_avg = []
        
        # loop through time
        N = len(self.qflux)
        for k in np.arange(N):
        
            # get q(t)
            q = qflux[k]
            t = time [k]
        
            # compute weights
            gamma = (t - t0)/tau
            alpha = np.e**( - gamma)
            delta = q - qavg
        
            # update averages
            qavg = alpha * qavg + q * (1 - alpha)
            var_qavg = alpha * ( var_qavg + (1-alpha)* delta**2)
            t0 = t
        
            # save
            Q_avg.append(qavg)
            Var_Q_avg.append(var_qavg)

        self.Q_avg = Q_avg
        self.Var_Q_avg = Var_Q_avg

        return Q_avg[-1], Var_Q_avg[-1]

    def check_convergence(self, tau_list=[10,50,100], threshold=0.5):

        '''
        Runs the exponential moving average for a list of taus

        Decides whether to halt, by comparing neighboring taus.
        '''
        print("tau_list", tau_list)

        data = [ self.exponential_window_estimator(tau=tau) for tau in tau_list]

        avg,var = np.transpose(data)
        std = np.sqrt(var)

        check = std/avg < threshold
        print(check)
        print(avg)

        N = len(tau_list)
        for j in range(N-1):
            
            c0 = check[j]

            if c0:
               c1 = check[j+1]

               if c1:
                   
                   # execute halt command
                   run_name = self.fname[:-3]
                   cmd = f"touch {run_name}.stop"
                   print(cmd)
                   return
        


class VMEC_GX_geometry_module():

    # this class handles VMEC-GX Geometry .ing input files

    def __init__(self, engine,
                       f_sample    = 'gx-geometry-sample.ing',
                       tag         = 'default',
                       input_path  = 'gx-files/',
                       output_path = './'
                       ):

        self.engine = engine
        self.data = self.read(input_path + f_sample)

        self.output_path = output_path
        self.input_path  = input_path
        self.tag  = tag


    # this function is run at __init__
    #    it parses a sample GX_geometry.ing input file
    #    as a dictionary for future modifications
    def read(self,fin):

        with open(fin) as f:
            indata = f.readlines()

        data = {} # create dictionary
        for line in indata:

            # remove comments
            info = line.split('#')[0]

            # skip blanks
            if info.strip() == '':
                continue

            # parse
            key,val = info.split('=')
            key = key.strip()
            val = val.strip()

            # save
            data[key] = val

        return data


    def write(self, fname): # writes a .ing file

        # load
        data = self.data
        path = self.output_path
        
        # set spacing
        longest_key =  max( data.keys(), key=len) 
        N_space = len(longest_key) 

        # write
        fout = path + fname + '.ing'
        with open(fout,'w') as f:
            for pair in data.items():
                s = '  {:%i}  =  {}' % N_space
                #print(s.format(*pair))   # print to screen for debugging
                print(s.format(*pair), file=f)


    def set_vmec(self,wout, vmec_path='./', output_path='./'):

        # copy vmec output from vmec_path to output_path
        #cmd = 'cp {:}{:} {:}'.format(vmec_path, wout, output_path)
        #os.system(cmd)

        self.data['vmec_file'] = '"{:}"'.format(wout)
        self.data['out_path'] = '"{:}"'.format(output_path)
        self.data['vmec_path'] = '"{:}"'.format(vmec_path) 

    def init_radius(self,rho,r_idx):

        # set radius
        s = rho**2
        self.data['desired_normalized_toroidal_flux'] = s

        t_idx = self.engine.t_idx
        file_tag = f"vmec-t{t_idx}-r{r_idx}"
        self.data['file_tag'] = f"\"{file_tag}\""

        # write input
        in_path  = self.input_path
        out_path = self.output_path
        fname = self.tag + '-psi-{:.2f}'.format(s)
        self.write(fname)
        print('  wrote .ing', out_path+fname)

        # run
        system = os.environ['GK_SYSTEM']
        cmd = ['{:}convert_VMEC_to_GX'.format(in_path),  out_path+fname]
        if system == 'traverse':
            cmd = ['convert_VMEC_to_GX',  out_path+fname]

        f_log = out_path + fname + '.log'
        with open(f_log, 'w') as fp:
            subprocess.call(cmd,stdout=fp)

        f_geometry = f"gx_geo_{file_tag}.nc"
        return f_geometry


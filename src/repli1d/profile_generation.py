import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import interp1d
import numpy as np


# plt.figure()

def generate_slow_background(x, scale=500, npoints=1000):
    kernel = 1.0 * RBF(length_scale=scale)

    X_ = np.linspace(0, x[-1], npoints)
    #print(X_.shape)
    gp_bg = GaussianProcessRegressor(kernel=kernel)
    y_mean, y_std = gp_bg.predict(X_[:, np.newaxis], return_std=True)

    y_samples = gp_bg.sample_y(X_[:, np.newaxis], 1)
    #y_samples -= np.min(y_s)
    #th=0.5
    #y_samples[y_samples <= -th] = -th
    #y_samples += th + 1e-7
    #print(y_samples.shape)
    f2 = interp1d(X_, y_samples[::, 0], kind='cubic', bounds_error=False)
    res = f2(np.arange(len(x)))
    #res[res<=0] = 0
    res -= np.min(res)
    return res


from scipy import signal


def signal_shape_gauss(size):
    s = signal.gaussian(size, std=size / 3)
    # print(np.max(s))
    s /= np.max(s)
    return s


def generate_gene_signal(w=[2, 30, 2], h=[10, 2],
                         end=False, clean_middle=False, gaussian=True,
                         clean=0, length_increase=5, orientation=0):
    signal_value = np.ones(np.sum(w))

    if gaussian:
        signal_shape = signal_shape_gauss
    else:
        signal_shape = lambda size: np.ones(size, dtype=np.float)

    signal_value[:w[0]] = signal_shape(w[0]) * h[0]
    if end:
        signal_value[-w[2]:] = signal_shape(w[2]) * h[1]

    if clean == 1:
        signal_value[w[0]:-w[2]] = 0

    if clean == 2:
        r = np.arange(len(signal_value[w[0]:-w[2]]))[::-1]
        signal_value[w[0]:-w[2]] = signal_value[w[0]:-w[2]] * np.exp(-r / length_increase)

    if orientation:
        return signal_value
    else:
        return signal_value[::-1]

def generate_group_genes(signal_v,coverage=0.7,value_noise=0.01):
    background = signal_v
    gene_per_group=20
    ngroup = len(background)/((2+30+2)*gene_per_group) * coverage
    ngroup = int(ngroup)
    height = 15
    total_value_group = ngroup * gene_per_group * (height/2 * 2)
    background /= np.mean(background)
    background *= (total_value_group * value_noise) / len(background)
    print("Ngroup",ngroup)
    print(np.sum(background))
    list_groups = np.concatenate((np.random.poisson(gene_per_group,ngroup),np.random.randint(1,3,max(ngroup//4,1))))
    ngroup = len(list_groups)
    for id_group,n in enumerate(np.random.choice(list_groups,replace=False,size=ngroup)):
        starts_w = np.random.poisson(2,n) +1 #To avaid empty group
        sizes_w = np.random.poisson(30,n)+1
        ends_w = np.random.poisson(2,n)+1
        add_ends = np.random.randint(0,2,n)
        cleans = np.random.randint(0,3,n)
        orientations = np.random.randint(0,2,n)
        length_increases = np.random.poisson(20,n)
        hs = np.random.poisson(height/2,n)+1
        he = np.random.poisson(height//4,n)+1




        tot_size = np.sum(starts_w)+np.sum(sizes_w)+np.sum(ends_w)
        tot_size *= 1+1*np.random.rand()
        tot_size = int(tot_size)
        #print(tot_size)
        #select pos of zone:
        if tot_size>len(background):
            continue
        delta = len(background) / (ngroup + 1)
        x=id_group * delta + np.random.randint(delta)
        if x-tot_size > len(background):
            continue

        pos = int(x)

        for start,size,end,add_end,clean,orientation,length_increase,hsi,hei in zip(starts_w,sizes_w,ends_w,add_ends,cleans,orientations,length_increases,hs,he):
            #print(clean)
            #print(start,size,end,add_end,clean,orientation,length_increase,h)
            new_gene =  generate_gene_signal(w=[start,size,end],h=[hsi,hei],
                                                                       end=add_end,clean=clean,
                                                                       orientation=orientation,
                                                                      length_increase=length_increase)
            if len(new_gene) == len( background[pos:pos+start+size+end]):
                background [pos:pos+start+size+end] = new_gene


            plus = np.random.poisson(30,1)[0]
            if np.random.randint(0,10,1)[0] == 0:
                plus = np.random.randint(1,2)
            pos = int(pos +plus)

    #Large bump
    for i in np.random.randint(31,len(background)-31,15):
        #print(i)
        size = np.random.randint(20,30)
        #print("Adding")
        new_gene = signal_shape_gauss(2*size) * 4*np.mean(background) *np.random.rand()
        if len(new_gene) == len(background[i-size:i+size]):
            background[i-size:i+size] *= new_gene
    print("End",np.sum(background))
    return background
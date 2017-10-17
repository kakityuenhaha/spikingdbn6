#coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pyNN.spiNNaker as p
import matplotlib.cm as cm
#import relu_utils as alg
#import spiking_relu as sr
import random
import pickle
#import mnist_utils as mu
import os.path
import scipy.io as sio
#import mnist_utils as mu
import random
#import relu_utils as alg
import copy
#import Debug_util as Du
import math
#import sys
#%%
#USAGE: spiking_dbn.py scaled_weight b10_epoc5
def matrix_times(m, n):
    m_matrix = np.transpose(np.tile(m,(len(n), 1)))
    n_matrix = np.tile(n,(len(m), 1))
    return m_matrix*n_matrix
    

def plot_digit(img_raw):
    #img_raw = np.uint8(img_raw)
    plt.figure(figsize=(5,5))
    im = plt.imshow(np.reshape(img_raw,(28,28)), cmap=cm.gray_r,interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter
    #random.expovariate(rateParameter)
def poisson_generator(rate, t_start, t_stop):
    poisson_train = []
    if rate > 0:
        next_isi = nextTime(rate)*1000.
        last_time = next_isi + t_start
        while last_time  < t_stop:
            poisson_train.append(last_time)
            next_isi = nextTime(rate)*1000.
            last_time += next_isi
    return poisson_train




def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
         pickle.dump(obj, f)
        #pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
def init_para(vis_num, hid_num, eta):
    para = {}
    para['h_num'] = hid_num
    para['v_num'] = vis_num
    para['eta'] = eta
    w = np.random.normal(0,0.01, vis_num*hid_num)
    para['w'] = w.reshape((vis_num,hid_num))
    return para


    
def update_batch_cd1(para, data_v):
    eta = para['eta']
    max_bsize = data_v.shape[0]
    data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    
    pos_delta_w = np.zeros((para['v_num'], para['h_num']))
    neg_delta_w = np.zeros((para['v_num'], para['h_num']))
    for i in range(max_bsize):
        pos_delta_w += matrix_times(data_v[i], data_h[i])
        neg_delta_w += matrix_times(gibbs_v[i], gibbs_h[i])    
    delta_w_pos = eta * pos_delta_w/np.float(max_bsize)
    delta_w_neg = eta * neg_delta_w/np.float(max_bsize)
    para['w'] += delta_w_pos
    para['w'] -= delta_w_neg
    #print delta_w_pos.max(), delta_w_neg.max()
    return para
    
def ReLU(data, weight):
    sum_data = np.dot(data, weight) # shapes (0,) and (784,500) not aligned: 0 (dim 0) != 784 (dim 0)
    sum_data[sum_data < 0] = 0
    return sum_data

def sampling(para, data_v):
    w = para['w']
    h0 = ReLU(data_v, w)
    v1 = ReLU(h0, w.transpose())
    h1 = ReLU(v1, w)
    return h0, v1, h1
    
def init_label_dbn(train_data, label_data, nodes, eta=1e-3, batch_size=10, epoc=5):
    if train_data.shape[1] != nodes[0]:
        print 'Dimention of train_data has to equal to the input layer size.'
        exit()
    elif label_data.shape[1] != nodes[-1]:
        print 'Dimention of label_data has to equal to the output layer size.'
        exit()
    elif train_data.shape[0] != label_data.shape[0]:
        print 'The amount of data and label should be the same.'
        exit()
    dbnet = {}
    dbnet['train_x'] = train_data
    dbnet['train_y'] = label_data
    dbnet['nodes'] = nodes
    dbnet['batch_size'] = batch_size
    dbnet['epoc'] = epoc
    
    para_list = []
    for i in range(len(nodes) - 3):   #bottom up
        para_list.append(init_para(nodes[i], nodes[i+1], eta))
    para_top = init_para(nodes[-3] + nodes[-1], nodes[-2], eta)
    para_top['label_n'] = 10
    dbnet['layer'] = para_list
    dbnet['top'] = para_top
    
    return dbnet

def RBM_train(para, epoc, batch_size, train_data):
    train_num = train_data.shape[0]
    for iteration in range(epoc):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            data_v = train_data[k:k+max_bsize]
            para = update_batch_cd1(para, data_v)
    return para

def greedy_train(dbnet):
    batch_size = dbnet['batch_size']
    train_size = dbnet['train_x'].shape[0]
    drop_out = 0.5
    train_index = np.random.choice(train_size, int(train_size*drop_out), replace=False)
    train_data = dbnet['train_x'][train_index]
    train_label = dbnet['train_y'][train_index]
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i] = RBM_train(dbnet['layer'][i], dbnet['epoc'], batch_size, train_data)
        train_data = ReLU(train_data, dbnet['layer'][i]['w'])
    train_data = np.append(train_data, train_label, axis=1) # if the row numbers of train_data and train_label are the same, axis equals 1, else axis equals 0;  
    dbnet['top'] = RBM_train(dbnet['top'], dbnet['epoc'], batch_size, train_data)
    return dbnet
    

def update_unbound_w(w_up, w_down, d_vis):
    bsize = d_vis.shape[0]
    delta_w = 0
    d_hid = ReLU(d_vis, w_up)
    g_vis = ReLU(d_hid, w_down)
    for ib in range(bsize):
        delta_w += matrix_times(d_hid[ib], d_vis[ib]-g_vis[ib])
    delta_w /= np.float(bsize)
    return delta_w, d_hid
    
def fine_train(dbnet):
    batch_size = dbnet['batch_size']
    train_data = dbnet['train_x']
    train_num = train_data.shape[0]
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i]['w_up'] = dbnet['layer'][i]['w']
        dbnet['layer'][i]['w_down'] = np.transpose(dbnet['layer'][i]['w'])
    for iteration in range(dbnet['epoc']):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            d_vis = train_data[k:k+max_bsize]
            label = dbnet['train_y'][k:k+max_bsize]
            #up
            for i in range(len(dbnet['layer'])):   #bottom up
                delta_w, d_vis = update_unbound_w(dbnet['layer'][i]['w_up'], dbnet['layer'][i]['w_down'], d_vis)
                dbnet['layer'][i]['w_down'] += dbnet['layer'][i]['eta'] * delta_w
            #top
            d_vis = np.append(d_vis, label, axis=1)
            dbnet['top'] = update_batch_cd1(dbnet['top'], d_vis)
            d_hid, g_vis, g_hid = sampling(dbnet['top'], d_vis)
            d_vis = g_vis[:, :dbnet['top']['v_num'] - dbnet['top']['label_n']]
            #down
            for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
                delta_w, d_vis = update_unbound_w(dbnet['layer'][i]['w_down'], dbnet['layer'][i]['w_up'], d_vis)
                dbnet['layer'][i]['w_up'] += dbnet['layer'][i]['eta'] * delta_w
    return dbnet

def dbn_recon(dbnet, test):
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w_up'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    temp = np.append(temp, label, axis=0)
    
    temp = ReLU(temp, dbnet['top']['w'])
    temp = ReLU(temp, np.transpose(dbnet['top']['w']))
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
        temp = ReLU(temp, dbnet['layer'][i]['w_down'])
    recon = temp
    plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon

def greedy_recon(dbnet, test):
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    temp = np.append(temp, label, axis=0) # here temp:(500,), label(10,)
    temp = ReLU(temp, dbnet['top']['w'])
    temp = ReLU(temp, np.transpose(dbnet['top']['w']))
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
        temp = ReLU(temp, np.transpose(dbnet['layer'][i]['w']))
    recon = temp
    plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon
    
def test_label_data(dbnet, test_data, test_label):
    dbnet['test_x'] = test_data
    dbnet['test_y'] = test_label
    return dbnet

def dbn_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w_up'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result

def dbn_greedy_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n'] # 100
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])  # 9999,400
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result




#w_listf = sys.argv[1]
#dbn_f = sys.argv[2]
w_listf = 'scaled_weight'
#dbn_f = 'special'
#dbnet = alg.load_dict(dbn_f)
#import scipy.io as sio
tmp_x=np.double(sio.loadmat('mnist_uint8.mat')['test_x'])
#tmp_x=np.transpose(tmp_x,(2,0,1))
#tmp_x=np.reshape(tmp_x,(tmp_x.shape[0],28*28,),order='F')


tmp_y=np.double(sio.loadmat('mnist_uint8.mat')['test_y'])
tmp_y=np.argmax(tmp_y,axis=1)
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
                   
if os.path.isfile('%s.pkl'%w_listf):
    scaled_w = load_dict(w_listf)
    w = scaled_w['w']
    k = scaled_w['k']
    x0 = scaled_w['x0']
    y0 = scaled_w['y0']
    print 'found w_list file'
#else:
#    w, k, x0, y0 = sr.w_adjust(dbnet, cell_params_lif)
#    scaled_w = {}
#    scaled_w['w'] = w
#    scaled_w['k'] = k
#    scaled_w['x0'] = x0
#    scaled_w['y0'] = y0
#    alg.save_dict(scaled_w, w_listf)

num_test = 100
random.seed(0)
dur_test = 1000
silence = 200
#test_x = dbnet['test_x']
#result_list = np.zeros((test_x.shape[0], 2))
count = 0

    #result_list[offset:offset+num_test, 0] = r
    #result_list[offset:offset+num_test, 1] = (result_list[offset:offset+num_test, 0] == dbnet['test_y'][offset:offset+num_test]).astype(int)
    #index = np.where(spike_count.max(axis=0)==0)[0]
    #result_list[offset+index, 1] = -1
    #print spike_count, np.argmax(spike_count, axis=0), dbnet['test_y'][:10]   
    #np.save('result_list1', result_list)

#import numpy as np
#import matplotlib.pyplot as plt
#import pyNN.spiNNaker as p


#def plot_spikes(spikes, title):
#    fig, ax = plt.subplots()
#    ax.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
#    plt.show()

def get_train_data():
    dir_path = '../'
    file_name = dir_path + 'train-images.idx3-ubyte'
    print file_name
    
    f = open(file_name, "rb")
    magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
    train_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    train_x = np.reshape(train_x, (list_size,image_hight*image_width))
    f.close()
    
    file_name = dir_path + 'train-labels.idx1-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
    train_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    f.close()
    
    return np.double(train_x), np.double(train_y)

# get testing data
def get_test_data():
    dir_path = '../'
    file_name = dir_path + 't10k-images.idx3-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
    test_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    test_x = np.reshape(test_x, (list_size,image_hight*image_width))
    f.close()
    
    file_name = dir_path +  't10k-labels.idx1-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
    test_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    f.close()
    
    return np.double(test_x), np.double(test_y)
    

#def nextTime(rateParameter):
#    return -math.log(1.0 - random.random()) / rateParameter
#    #random.expovariate(rateParameter)
#def poisson_generator(rate, t_start, t_stop):
#    poisson_train = []
#    if rate > 0:
#        next_isi = nextTime(rate)*1000.
#        last_time = next_isi + t_start
        #print t_stop
        #print last_time
#        while last_time  < t_stop:
#            poisson_train.append(last_time)
#            next_isi = nextTime(rate)*1000.
#            last_time += next_isi

#    return poisson_train

def mnist_poisson_gen(image_list, image_height, image_width, max_freq, duration, silence):
    #print image_list.shape[1]
    #print max_freq
    if max_freq > 0:
        for i in range(image_list.shape[0]):
            image_list[i] = image_list[i]/sum(image_list[i])*max_freq
            #print image_list[i]
    spike_source_data = [[] for i in range(image_height*image_width)]
    
    for i in range(image_list.shape[0]):
        #print i
        t_start = i*(duration+silence)
        t_stop = t_start+duration
        for j in range(image_height*image_width):
            spikes = poisson_generator(image_list[i][j], t_start, t_stop)
            if spikes != []:
                spike_source_data[j].extend(spikes)
    #print spike_source_data
    return spike_source_data

def transf(k, x0, y0, curr):
    rate = k*(curr-x0) + y0
    rate[rate<0] = 0
    return rate
    
def rev_transf(k, x0, y0, rate):
    curr = (rate - y0) / k + x0
    return curr

'''
sim = sys.argv[1]
if sim == 'nest':
    import pyNN.nest as p
elif sim == 'spin':
    import spynnaker.pyNN as p
else:
    sys.exit()
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,   # 2.0
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
'''

def estimate_kb(cell_params_lif):
    cell_para = copy.deepcopy(cell_params_lif)
    random.seed(0)
    p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)
    run_s = 10.
    runtime = 1000. * run_s
    max_rate = 1000.
    ee_connector = p.OneToOneConnector(weights=1.0, delays=2.0)    


    pop_list = []
    pop_output = []
    pop_source = []
    x = np.arange(0., 1.01, 0.1)
    count = 0
    trail = 10

    for i in x:
        for j in range(trail): #trails for average
            pop_output.append(p.Population(1, p.IF_curr_exp, cell_para))
            poisson_spikes = poisson_generator(i*max_rate, 0, runtime)
            pop_source.append( p.Population(1, p.SpikeSourceArray, {'spike_times' : poisson_spikes}) )
            p.Projection(pop_source[count], pop_output[count], ee_connector, target='excitatory')
            pop_output[count].record()
            count += 1


    count = 0
    for i in x:
        cell_para['i_offset'] = i
        pop_list.append(p.Population(1, p.IF_curr_exp, cell_para))
        pop_list[count].record()
        count += 1
    pop_list[count-1].record_v()

    p.run(runtime)

    rate_I = np.zeros(count)
    rate_P = np.zeros(count)
    rate_P_max = np.zeros(count)
    rate_P_min = np.ones(count) * 1000.
    for i in range(count):
        spikes = pop_list[i].getSpikes(compatible_output=True)
        rate_I[i] = len(spikes)/run_s
        for j in range(trail):
            spikes = pop_output[i*trail+j].getSpikes(compatible_output=True)
            spike_num = len(spikes)/run_s
            rate_P[i] += spike_num
            if spike_num > rate_P_max[i]:
                rate_P_max[i] = spike_num
            if spike_num < rate_P_min[i]:
                rate_P_min[i] = spike_num
        rate_P[i] /= trail
    '''
    #plot_spikes(spikes, 'Current = 10. mA')
    plt.plot(x, rate_I, label='current',)
    plt.plot(x, rate_P, label='Poisson input')
    plt.fill_between(x, rate_P_min, rate_P_max, facecolor = 'green', alpha=0.3)
    '''
    x0 = np.where(rate_P>1.)[0][0]
    x1 = 4
    k = (rate_P[x1] - rate_P[x0])/(x[x1]-x[x0])
    '''
    plt.plot(x, k*(x-x[x0])+rate_P[x0], label='linear')
    plt.legend(loc='upper left', shadow=True)
    plt.grid('on')
    plt.show()
    '''
    p.end()
    return k, x[x0], rate_P[x0]

def w_adjust(dbnet, cell_para, SUM_rate=2000., lim_rate = 20.):
    #Du.set_trace()
    train_x = np.copy(dbnet['train_x'])
    k, x0, y0 = estimate_kb(cell_para)
    for i in range(train_x.shape[0]):
        train_x[i] = train_x[i] / sum(train_x[i]) * SUM_rate
    w_list = []
    for i in range(len(dbnet['layer'])):
        #w = np.copy(dbnet['layer'][i]['w_up'])
        w = np.copy(dbnet['layer'][i]['w'])
        scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
        w_list.append(w * scale)

    h_num = dbnet['top']['v_num'] - dbnet['top']['label_n']
    w = np.copy(dbnet['top']['w'][:h_num,:])
    scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
    w_list.append(w * scale)
    w = np.copy(np.transpose(dbnet['top']['w'][h_num:,:]))
    scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
    w_list.append(w * scale)
    
    return w_list, k, x0, y0

def scale_to_spike(train_x, w, k, x0, y0, lim_rate):
    curr = ReLU(train_x, w)
    mean = np.mean(curr)
    #std = np.std(curr)
    #x_lim = rev_transf(k, x0, y0, lim_rate) 
    #scale = x_lim * 1000. / (mean + 3 * std)
    x_mid = rev_transf(k, x0, y0, lim_rate) 
    scale = x_mid * 1000. / mean
    curr *= scale/1000.
    out_rate = transf(k, x0, y0, curr)
    '''
    count, edges = np.histogram(curr)
    width = edges[1] - edges[0]
    plt.bar(edges[:-1], count, width=width)
    plt.show()
    '''
    return scale, out_rate
    
def gen_spike_source(data, SUM_rate=2000., dur_test=1000, silence=200):
    input_size = 28
    spike_source_data = mnist_poisson_gen(data, input_size, input_size, SUM_rate, dur_test, silence)
    return spike_source_data
    
def run_test(w_list, cell_para, spike_source_data):
    #Du.set_trace()
    pop_list = []
    p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
    #input poisson layer
    input_size = w_list[0].shape[0]
    #print w_list[0].shape[0]
    #print w_list[1].shape[0]
    
    #list=[]
    #for j in range(input_size):
     #   list.append(spike_source_data[j])
    #pop_in = p.Population(input_size, p.SpikeSourceArray, {'spike_times' :list})
    
    #pop_list.append(pop_in)
    
    #for j in range(input_size):
        #pop_in[j].spike_times = spike_source_data[j]
    
    pop_in = p.Population(input_size, p.SpikeSourceArray, {'spike_times' : []})
    for j in range(input_size):
        pop_in[j].spike_times = spike_source_data[j]
    pop_list.append(pop_in)
    
    #count =0
    #print w_list[0].shape[0]
    for w in w_list:
        input_size = w.shape[0]
        #count = count+1
        #print count
        output_size = w.shape[1]
        #pos_w = np.copy(w)
        #pos_w[pos_w < 0] = 0
        #neg_w = np.copy(w)
        #neg_w[neg_w > 0] = 0
        conn_list_exci=[]
        conn_list_inhi=[]
    #k_size=in_size-out_size+1
        for x_ind in range(input_size):
            for y_ind in range(output_size):
                weights = w[x_ind][y_ind]
                #for i in range(w.shape[1]):
                if weights>0:
                    conn_list_exci.append((x_ind,y_ind,weights,1.))
                elif weights<0:
                    conn_list_inhi.append((x_ind,y_ind,weights,1.))
        #print output_size
        pop_out = p.Population(output_size, p.IF_curr_exp, cell_para)
        if len(conn_list_exci)>0:
            p.Projection(pop_in,pop_out,p.FromListConnector(conn_list_exci),target='excitatory')
        if len(conn_list_inhi)>0:
            p.Projection(pop_in,pop_out,p.FromListConnector(conn_list_inhi),target='inhibitory')
        #p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = pos_w), target='excitatory')
        #p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = neg_w), target='inhibitory')
        pop_list.append(pop_out)
        pop_in = pop_out

    pop_out.record()
    run_time = np.ceil(np.max(spike_source_data)[0]/1000.)*1000
    #print run_time
    p.run(run_time)
    spikes = pop_out.getSpikes(compatible_output=True)
    return spikes

def counter(data, left_edge, dur):
    count = []
    for l in left_edge:
        temp = np.where((data >= l) & (data < l+dur))
        count.append(temp[0].shape[0])
    return count
    
#def test_sdbn():

#    return result

for offset in range(0, tmp_x.shape[0], num_test):
#for offset in range(0, 1000, num_test):
    print offset
    test = tmp_x[offset:(offset+num_test), :]
    test=test*120.
    spike_source_data = gen_spike_source(test)                
    #spikes = run_test(w, cell_params_lif, spike_source_data)
    #Du.set_trace()
    pop_list = []
    p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
    #input poisson layer
    input_size = w[0].shape[0]
    #print w_list[0].shape[0]
    #print w_list[1].shape[0]
    
    #list1=[]
    #for j in range(input_size):
    #    list1.append(spike_source_data[j])
    #pop_in = p.Population(input_size, p.SpikeSourceArray, {'spike_times' :list1})
    
    #pop_list.append(pop_in)
    
    #for j in range(input_size):
        #pop_in[j].spike_times = spike_source_data[j]
    
    pop_in = p.Population(input_size, p.SpikeSourceArray, {'spike_times' : []})
    for j in range(input_size):
        pop_in[j].spike_times = spike_source_data[j]
    pop_list.append(pop_in)
    
    #count =0
    #print w_list[0].shape[0]
    for w1 in w:
        input_size = w1.shape[0]
        #count = count+1
        #print count
        output_size = w1.shape[1]
        pos_w = np.copy(w1)
        pos_w[pos_w < 0] = 0
        neg_w = np.copy(w1)
        neg_w[neg_w > 0] = 0
        #conn_list_exci=[]
        #conn_list_inhi=[]
    #k_size=in_size-out_size+1
        #for x_ind in range(input_size):
            #for y_ind in range(output_size):
                #weights = w1[x_ind][y_ind]
                #weights = w1[x_ind]
                #for i in range(w.shape[1]):
                #if weights>0:
                    #conn_list_exci.append((x_ind,y_ind,weights,1.))
                #elif weights<0:
                    #conn_list_inhi.append((x_ind,y_ind,weights,1.))
        #print output_size
        pop_out = p.Population(output_size, p.IF_curr_exp, cell_params_lif)
        #if len(conn_list_exci)>0:
            #p.Projection(pop_in,pop_out,p.FromListConnector(conn_list_exci),target='excitatory')
        #if len(conn_list_inhi)>0:
            #p.Projection(pop_in,pop_out,p.FromListConnector(conn_list_inhi),target='inhibitory')
        p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = pos_w), target='excitatory')
        p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = neg_w), target='inhibitory')
        pop_list.append(pop_out)
        pop_in = pop_out

    pop_out.record()
    run_time = np.ceil(np.max(spike_source_data)[0]/1000.)*1000
    #print run_time
    p.run(run_time)
    spikes = pop_out.getSpikes(compatible_output=True)
    
    
    
    spike_count = list()

    for i in range(w[-1].shape[1]):
        index_i = np.where(spikes[:,0] == i)
        spike_train = spikes[index_i, 1]
        temp = counter(spike_train, range(0, (dur_test+silence)*num_test,dur_test+silence), dur_test)
        spike_count.append(temp)
    spike_count = np.array(spike_count)/(dur_test / 1000.)
    r = np.argmax(spike_count, axis=0)
    correct = np.sum(r == tmp_y[offset:offset+num_test]).astype(int) #- len(np.where(spike_count.max(axis=0)==0)[0])
    print 'correct number'#
    print correct
    count = count + correct
print 'count'
print count
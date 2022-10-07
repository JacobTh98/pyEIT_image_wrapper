import numpy as np
import matplotlib.pyplot as plt
import pyeit.mesh as mesh
import pyeit.eit.jac as jac
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.fem import Forward
from tqdm import tqdm
import os
import glob
import math
import cv2
import requests
import pandas as pd
#import seaborn as sns

fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 25,
        }
plt.rcParams.update({'font.family': 'serif'})
el_pos = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]



def telegram_KI_bot(bot_message):
    bot_token = '5303222968:AAFvfDsh1mLDESs7tFfwJlObW-RqnJRXufQ'
    bot_chatID = '5140274444'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)

def set_perm_cube(mesh_obj, anomaly):
    """Erstellen eines quaders"""
    pts = mesh_obj["element"]
    tri = mesh_obj["node"]
    perm = mesh_obj["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    n = np.size(mesh_obj["perm"])
    # reset background if needed
    perm = np.ones(n)
    # change dtype to 'complex' for complex-valued permittivity
    if anomaly is not None:
        for attr in anomaly:
            if np.iscomplex(attr["perm"]):
                perm = perm.astype("complex")
                break
    # assign anomaly values (for elements in regions)
    if anomaly is not None:
        for _, attr in enumerate(anomaly):
            d = attr["d"]  # ist d = 0.2
            index = (np.sqrt((tri_centers[:, 1] - attr["y"])**2) <
                     d) & (np.sqrt((tri_centers[:, 0] - attr["x"])**2) < d)
            perm[index] = attr["perm"]  # Zuweisen von 10
    mesh_new = {"node": tri, "element": pts, "perm": perm}
    return mesh_new


def set_perm_triangle(mesh_obj, anomaly):
    pts = mesh_obj["element"]
    tri = mesh_obj["node"]
    perm = mesh_obj["perm"].copy()
    #tri_centers = np.mean(tri[pts], axis=1)
    global_nodes = np.mean(tri[pts], axis=1)
    n = np.size(mesh_obj["perm"])
    # reset background if needed
    perm = np.ones(n)
    # change dtype to 'complex' for complex-valued permittivity
    if anomaly is not None:
        for attr in anomaly:
            if np.iscomplex(attr["perm"]):
                perm = perm.astype("complex")
                break
    # assign anomaly values (for elements in regions)
    if anomaly is not None:
        for _, attr in enumerate(anomaly):
            d = attr["d"]  # ist d = 0.2
            #attr["y"] = attr["y"]+d
            #index = (np.sqrt((tri_centers[:, 0] - attr["x"])**2) < d) & (np.sqrt((tri_centers[:, 1] - attr["y"])**2) > d)
            x_len = y_len = np.arange(-1, 1, 0.01)
            for i in range(global_nodes.shape[0]):
                x, y = 0, 1
                roe = 100
                m = 2
                n = attr["x"]
                if m*(np.round(global_nodes[i][x]*roe)/roe) < -(np.round(global_nodes[i][y]*roe)/roe - attr["x"] - attr["y"] - d) and m*(np.round(global_nodes[i][x]*roe)/roe) > (np.round(global_nodes[i][y]*roe)/roe + attr["x"] - attr["y"] - d) and global_nodes[i][y] >= attr["y"]-d:
                    perm[i] = attr["perm"]

    mesh_new = {"node": tri, "element": pts, "perm": perm}
    return mesh_new

def generate_mesh(n_el=16, x=0, y=0, d=0.2, perm=10):
    '''generate mesh'''
    def _fd(pts):
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 1.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0
    mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)
    return mesh_obj

def generate_mesh_triangle(n_el=16, x=0, y=0, d=0.2, perm=10):
    '''generate mesh for hip-stem implant - two centric regions with different permittivities'''
    def _fd(pts):
        # distance function for ellipse
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 1.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0

    # create mesh
    mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)
    # change permittivities
    anomaly = [{'x': x, 'y': y, 'd': d, 'perm': perm}]
    mesh_obj = set_perm_triangle(mesh_obj, anomaly=anomaly)
    return mesh_obj, el_pos


def generate_mesh_circle(n_el=16, x=-0.2, y=0.3, d=0.2, perm=10):
    '''generate mesh for hip-stem implant - two centric regions with different permittivities'''

    def _fd(pts):
        # distance function for ellipse
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 1.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0

    # create mesh
    mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)

    # change permittivities
    anomaly = [{'x': x, 'y': y, 'd': d, 'perm': perm}]
    # HIER RANDOM
    mesh_obj = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1)
    #mesh_obj = set_perm_cube(mesh_obj, anomaly=anomaly)
    return mesh_obj, el_pos


def generate_mesh_cube(n_el=16, x=-0.2, y=0.3, d=0.2, perm=10):
    '''generate mesh for hip-stem implant - two centric regions with different permittivities'''
    def _fd(pts):
        # distance function for ellipse
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 1.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0

    # create mesh
    mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)

    # change permittivities
    anomaly = [{'x': x, 'y': y, 'd': d, 'perm': perm}]
    # HIER RANDOM
    #mesh_obj = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1)
    mesh_obj = set_perm_cube(mesh_obj, anomaly=anomaly)
    return mesh_obj, el_pos


def plot_mesh(mesh_obj, el_pos=el_pos):
    '''plot the mesh and permittivities'''
    plt.style.use('default')
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = pts[:, 0], pts[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tripcolor(x, y, tri, np.real(mesh_obj['perm']),
                 edgecolors='k', shading='flat', alpha=0.5,
                 cmap=plt.cm.viridis)
    # draw electrodes
    ax.plot(x[el_pos], y[el_pos], 'ro')
    for i, e in enumerate(el_pos):
        ax.text(x[e], y[e], str(i+1), size=12)

    ax.set_title(r'mesh', fontdict=fontdict)
    ax.set_aspect('equal')
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)


def plot_reconstruction(mesh_obj, ds):
    '''plot the EIT reconstruction'''
    plt.style.use('default')
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = pts[:, 0], pts[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.tripcolor(x, y, tri, ds, shading='flat')
    for i, e in enumerate(el_pos):
        ax.annotate(str(i+1), xy=(x[e], y[e]), color='r')
    # fig.colorbar(im)

    ax.set_title(r'reconstruction', fontdict=fontdict)
    ax.set_aspect('equal')
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)


def pol2cart(r, a):
    x, y = r*np.cos(math.radians(a)), r*np.sin(math.radians(a))
    return x, y


def compute_electrode_signals(mesh_obj, el_pos, n_el, el_dist, SNR=None):
    '''compute electrode signals for given mesh'''

    ex_mat = eit_scan_lines(n_el, el_dist)
    fwd = Forward(mesh_obj, el_pos)
    f = fwd.solve_eit(ex_mat, step=1, perm=mesh_obj['perm'])

    if SNR is None:
        electrode_signals = f.v
    else:
        sigma_n2 = np.var(f.v) * 10**(SNR/10)
        n = np.sqrt(sigma_n2)*np.random.normal(size=f.v.shape)
        electrode_signals = f.v + n

    return ex_mat, electrode_signals


def compute_reconstruction(mesh_obj, el_pos, ex_mat, electrode_signals):
    '''compute EIT reconstruction'''

    eit = jac.JAC(mesh_obj, el_pos, ex_mat, step=1, perm=1.0, parser='std')
    eit.setup(p=0.25, lamb=1.0, method='lm')
    ds = eit.gn(electrode_signals, lamb_decay=0.1,
                lamb_min=1e-5, maxiter=15, verbose=True)

    return ds


def plot_overview(fpath, size=4, save=False):
    '''plot random meshes of a dataset'''
    plt.style.use('default')
    plt.rcParams.update({'font.size': 0})
    ld = np.random.randint(0, len(os.listdir(fpath)), size=size)
    
    el_pos = np.arange(16)
    fig = plt.figure()
    fig, ax = plt.subplots(size//2, size//2, sharex=True, sharey=True)
    for i in range(len(ld)//2):
        for j in range(len(ld)//2):
            tmp = np.load(
                fpath+'sample_{0:06d}'.format(ld[i+j])+'.npz', allow_pickle=True)
            mesh_obj = tmp['mesh_obj'].tolist()
            pts = mesh_obj['node']
            tri = mesh_obj['element']
            x, y = pts[:, 0], pts[:, 1]

            ax[i, j].tripcolor(x, y, tri, np.real(mesh_obj['perm']),
                               edgecolors='k', shading='flat', alpha=1.,
                               cmap=plt.cm.viridis)
            ax[i, j].set_aspect('equal')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_ylim([-1.1, 1.1])
            ax[i, j].set_xlim([-1.1, 1.1])

    fig.set_size_inches(10, 10)
    plt.tight_layout()
    if save:
        fname = 'overview_samples'+fpath[len(fpath)-4:-1]+'.pdf'
        plt.savefig(fname)
        print("Gespeichert als:\t", fname)
    plt.show()
    plt.rcParams.update({'font.size': 15})
    
def plot_overview_reconst(fpath, size=4, save=False, start_idx=0):
    '''plot reconstruction overview'''
    plt.style.use('default')
    plt.rcParams.update({'font.size': 0})
    ld = np.random.randint(start_idx, len(os.listdir(fpath)), size=size)
    
    el_pos = np.arange(16)
    fig = plt.figure()
    fig, ax = plt.subplots(size//2, size//2, sharex=True, sharey=True)
    for i in range(len(ld)//2):
        for j in range(len(ld)//2):
            tmp = np.load(
                fpath+'sample_{0:06d}'.format(ld[i+j])+'.npz', allow_pickle=True)
            mesh_obj = tmp['mesh_obj'].tolist()
            pts = mesh_obj['node']
            tri = mesh_obj['element']
            x, y = pts[:, 0], pts[:, 1]
            
            el_signals = np.expand_dims(np.expand_dims(tmp['electrode_signals'],axis=0),axis=-1)
            z = mapper.predict(el_signals)
            X_pred = vae.decoder.predict(z)
            
            ax[i, j].tripcolor(x, y, tri, np.real(mesh_obj['perm']),
                               edgecolors='k', shading='flat', alpha=1,
                               cmap=plt.cm.viridis)
            ax[i, j].tripcolor(x, y, tri, np.squeeze(X_pred),
                               edgecolors='k', shading='flat', alpha=0.5,
                               cmap=plt.cm.viridis)
            
            ax[i, j].set_aspect('equal')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_ylim([-1.1, 1.1])
            ax[i, j].set_xlim([-1.1, 1.1])

    fig.set_size_inches(10, 10)
    plt.tight_layout()
    if save:
        fname = 'diff_reconstruction'+fpath[len(fpath)-4:-1]+'.pdf'
        plt.savefig(fname)
        print("Gespeichert als:\t", fname)
    plt.show()
    plt.rcParams.update({'font.size': 15})


def generate_two_objts(konstel, n_el=16, x_1=0.5, y_1=0, d_1=0.2, perm_1=10, x_2=-0.5, y_2=0, d_2=0.2, perm_2=10):
    '''generate mesh for hip-stem implant - two centric regions with different permittivities'''
    def _fd(pts):
        # distance function for ellipse
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 1.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0

    # create mesh
    mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)
    # change permittivities
    anomaly = [{'x': x_1, 'y': y_1, 'd': d_1, 'perm': perm_1},
               {'x': x_2, 'y': y_2, 'd': d_2, 'perm': perm_2}]

    if konstel == 0:
        # Kreise
        mesh_obj = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1)
    elif konstel == 1:
        # Viereck und Kreis
        mesh_obj = mesh.set_perm(
            mesh_obj, anomaly=[anomaly[0]], background=0)
        mesh_obj_2 = set_perm_cube(mesh_obj, anomaly=[anomaly[1]])
        mesh_obj_2['perm'][mesh_obj_2['perm'] == 1] = 0
        mesh_obj['perm'] = mesh_obj_2['perm'] + mesh_obj['perm']
        mesh_obj['perm'][mesh_obj['perm'] == 0] = 1

    elif konstel == 2:
        # Kreis und Viereck
        mesh_obj = mesh.set_perm(
            mesh_obj, anomaly=[anomaly[1]], background=0)
        mesh_obj_2 = set_perm_cube(mesh_obj, anomaly=[anomaly[0]])
        mesh_obj_2['perm'][mesh_obj_2['perm'] == 1] = 0
        mesh_obj['perm'] = mesh_obj_2['perm'] + mesh_obj['perm']
        mesh_obj['perm'][mesh_obj['perm'] == 0] = 1
    elif konstel == 3:
        # Vierecke
        mesh_obj = set_perm_cube(mesh_obj, anomaly=anomaly)

    return mesh_obj, el_pos


def generate_three_obj():
    print("not implemented")
    return


""" Evaluation of Precision """
def delta_perm(path, skip=10):
    """Print the min and max perm of all samples."""
    perms = []
    fnames = os.listdir(path)
    for file in tqdm(fnames[::skip]):
        tmp = np.load(path+file,allow_pickle=True)['mesh_obj'].tolist()
        perms.append(tmp['perm'])
    perms = np.array(perms)
    print("Max Perm:\t",np.max(perms),"\n Min Perm:\t",np.min(perms[perms>1]))
    
    
def center_of_perm(mesh, plot=False):    
    """Calculates the center of permitivity"""
    pts = mesh["element"]
    tri = mesh["node"]
    perm = mesh["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_perm = np.array(mesh['perm'].tolist())
    perm_min = np.min(mesh_perm)
    ixds = np.array(np.where(mesh_perm>perm_min)[0]) #Indexes, wo perm > perm_min
    koords = tri_centers[ixds]
    x = round(np.mean(koords[:,0]),2)
    y = round(np.mean(koords[:,1]),2)
    print('x:',x,'y:',y)
    if plot:
        plt.figure(figsize=(5,5))
        plt.grid()
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        plt.scatter(tri_centers[:,0], tri_centers[:,1], color='grey')
        plt.scatter(koords[:,0], koords[:,1], color='orange')
        plt.scatter(x,y, color='red')
        
def evaluation_of_precision(mesh1,mesh2, plot=False):    
    """Calculates deviation of perm center and reconstructed perm space"""
    pts = mesh1["element"]
    tri = mesh1["node"]
    perm = mesh1["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_perm = np.array(mesh1['perm'].tolist())
    perm_min = np.min(mesh_perm)
    ixds = np.array(np.where(mesh_perm>perm_min)[0]) #Indexes, wo perm > perm_min
    koords = tri_centers[ixds]
    x = round(np.mean(koords[:,0]),3)
    y = round(np.mean(koords[:,1]),3)
    
    pts2 = mesh2["element"]
    tri2 = mesh2["node"]
    perm2 = mesh2["perm"].copy()
    tri_centers2 = np.mean(tri2[pts2], axis=1)
    mesh_perm2 = np.array(mesh2['perm'].tolist())
    perm_min2 = np.mean(mesh_perm2)+10*np.min(mesh_perm2)
    ixds2 = np.array(np.where(mesh_perm2>perm_min2)[0]) #Indexes, wo perm > perm_min
    koords2 = tri_centers2[ixds2]
    x2 = round(np.mean(koords2[:,0]),3)
    y2 = round(np.mean(koords2[:,1]),3)
    delta_x = x-x2
    delta_y = y-y2
    delta_perm = (len(koords[:,0])-len(koords2[:,0]))
    if plot:
        print('x:',x2,'y:',y2)  
        print("Δx:",delta_x,"Δy:",delta_y)
        print("ΔPerms:",delta_perm)
        plt.figure(figsize=(5,5))
        plt.rcParams.update({'font.family': 'Serif'})
        plt.grid()
        plt.xlim((-1.1,1.1))
        plt.ylim((-1.1,1.1))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(tri_centers[:,0], tri_centers[:,1], label=r'Objektleer', alpha=0.4, s =10)
        plt.scatter(koords[:,0], koords[:,1], color='green', label=r'Vorgabe')
        plt.scatter(koords2[:,0], koords2[:,1], color='orange', label=r'Rekonstruktion')
        plt.scatter(x,y,marker='x', color='red', s=50, label=r'$P_{mid}$ Vorgabe')
        plt.scatter(x2,y2, color='red', s=50, label=r'$P_{mid}$ Rekonstruktion')
        plt.tight_layout()
        plt.legend(loc='lower left')
    return delta_x,delta_y,delta_perm


def plot_deviations_x_y(Dict, save=False, fpath='',fname='x_y_deviation.pdf'):
    """Dict = {'x-Abweichung': X, 'y-Abweichung': Y, 'Perm':Perm}"""
    plt.rcParams.update({'font.family': 'Serif'})
    df = pd.DataFrame.from_dict(Dict)
    plt.figure(figsize=(7,7))
    sns.set(font_scale = 1.5, font='Serif')
    xlim = ylim = [-1, 1] 
    g=sns.jointplot(data=df, x='x-Abweichung',y='y-Abweichung', kind='kde', xlim =xlim ,ylim=ylim)
    g.plot_joint(sns.kdeplot,fill=True, levels=50, cmap="viridis")
    print("Durchschnittliche x-Abw:", round(np.mean(Dict['x-Abweichung']),2))
    print("Durchschnittliche y-Abw:", round(np.mean(Dict['y-Abweichung']),2))
    if save:
        plt.tight_layout()
        g.savefig(fpath+fname)
        
def plot_deviations_perm(Dict, save=False, fpath='', fname='perm_deviation.pdf',binwidth=10):
    plt.rcParams.update({'font.family': 'Serif'})
    """Dict = {'x-Abweichung': X, 'y-Abweichung': Y, 'Perm':Perm}"""
    df = pd.DataFrame.from_dict(Dict)
    plt.figure(figsize=(7,7))
    plt.autoscale()
    sns.set(font_scale = 2, font='Serif')
    p = sns.histplot(data=df,x='Perm',binwidth=binwidth, kde=True) #, kde=True
    #p.invert_xaxis()
    p.set_xlabel("Abweichende Elemente")
    p.set_ylabel("Anzahl")
    fig = p.get_figure()
    print('Mittlere Perm-Abweichung:',round(np.mean(Dict['Perm'])),'[FE]')
    st_fe = 2807
    print('Prozentuale Abweichung:',round((np.mean(Dict['Perm'])/st_fe)*100,2),'[%]')
    if save:
        plt.tight_layout()
        plt.savefig(fpath+fname)
        
def plot_losses_vae_mapper(spath, save=False):
    """Export Loss-plots of vae and mapper directly to input path."""
    loss_vae = np.load(spath+'vae_losses.npz', allow_pickle=True)
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'axes.grid': True})
    plt.rcParams.update({'figure.figsize': [15, 7]})
    plt.rcParams.update({'axes.grid': False})
    plt.rcParams.update({'axes.linewidth': 2.})
    plt.rcParams.update({'axes.labelsize': 'medium'})
    plt.rcParams.update({'axes.linewidth': 1.2})
    plt.rcParams.update({'font.family': 'serif'})
    plt.plot(loss_vae['loss'], label='Total loss', linewidth=3)
    plt.plot(loss_vae['reconstruction_loss'], label='Reconstruction loss', linewidth=3)
    plt.ylabel(r'Loss', fontdict=fontdict)
    plt.xlabel(r'Epoche', fontdict=fontdict)
    plt.xticks(np.arange(0,len(loss_vae['loss']),2))
    plt.legend()
    plt.grid(True, which='both')
    if save:
        plt.tight_layout()
        plt.savefig(spath+'model_loss_vae.pdf')
    plt.show()
    loss_mapper = np.load(spath+'mapper_losses.npz', allow_pickle=True)
    plt.plot(loss_mapper['loss'], linewidth=3)
    plt.ylabel(r'Loss', fontdict=fontdict)
    plt.xlabel(r'Epoche', fontdict=fontdict)
    plt.xticks(np.arange(0,len(loss_mapper['loss'])+1,len(loss_mapper['loss'])//10))
    plt.grid(True, which='both')
    #plt.show()
    if save:
        plt.tight_layout()
        plt.savefig(spath+'model_loss_mapper.pdf')
    #plt.style.use('default')
    
def ground_truth_template(objct='circle',r=0.5,phi=0,d=0.5):
    """
    input:  - single Object: 'circle', 'square', 'triangle'
            - radius
            - phi
            - d
    return: IMG (100x100 pixel picture)
    """
    def pol2cart(r, phi):
        x = r * np.cos(phi)+100
        y = r * np.sin(phi)+100
        return (int(x), int(y))

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def draw_circle(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)#*0 für die Rotation
        color = (1,0,0)
        thickness = -1
        IMG = cv2.circle(IMG, center_coordinates, int(d*100), color, thickness)
        return IMG

    def draw_square(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)#*0 für die Rotation
        start_point = (int(center_coordinates[0]-int(d*100)), center_coordinates[1]-int(d*100))
        end_point = (center_coordinates[0]+int(d*100), center_coordinates[1]+int(d*100))
        color = (1, 0, 0)
        thickness = -1
        IMG = cv2.rectangle(IMG, start_point, end_point, color, thickness)
        return IMG
    
    def draw_triangle(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)
        pt1 = (int(center_coordinates[0]), int(center_coordinates[1]-int((d)*100)))
        pt2 = (int(center_coordinates[0]+int((d)*100)), int(center_coordinates[1])+int((d)*100))
        pt3 = (int(center_coordinates[0]-int((d)*100)), int(center_coordinates[1])+int((d)*100))
        tri_edges = np.array( [pt1, pt2, pt3] )
        IMG = cv2.drawContours(IMG, [tri_edges], 0, (1,0,0), -1)
        return IMG
        
    r = int(r*100)
    angle = phi #Needed for rotation
    phi = math.radians(phi)# Grad in Rad

    if objct == 'circle':
        IMG = draw_circle(r,phi,d)
    if objct == 'square':
        IMG = draw_square(r,phi,d)
    if objct == 'triangle':
        IMG = draw_triangle(r,phi,d)
        
    IMG = rotate_image(IMG,angle)    
    return IMG
    
    
def groundtruth_IMG_based(IMG, set_perm = 10):
    """
    Input: IMG
    0 is set to Perm 1
    1 is meshed to Perm 10
    """
    X_Y=(np.array(np.where(IMG==1)))
    X = X_Y[1,:] -100
    Y = (X_Y[0,:] -100)*-1
    mesh_obj = generate_mesh()
    pts = mesh_obj["element"]
    tri = mesh_obj["node"]
    perm = mesh_obj["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_x = np.round(tri_centers[:,0]*100)
    mesh_y = np.round(tri_centers[:,1]*100)
    Perm = np.ones(tri_centers.shape[0])
    for i in range(len(X)):#IMG koordinaten
        for j in range(len(mesh_x)):# tri_centers
            if X[i] == mesh_x[j] and Y[i] == mesh_y[j]:
                Perm[j] = set_perm
    mesh_obj['perm'] = Perm
    return mesh_obj

def groundtruth_templated(objct='circle',r=0.5,phi=0,d=0.5):
    """
    input:  - single Object: 'circle', 'square', 'triangle'
            - radius
            - phi
            - d
    return: mesh_obj
    """
    obj_desc = {'objct': objct, 'r':r,'phi': phi,'d':d}
    IMG = ground_truth_template(**obj_desc)
    X_Y=(np.array(np.where(IMG==1)))
    X = X_Y[1,:] -100
    Y = (X_Y[0,:] -100)*-1
    mesh_obj = generate_mesh()
    pts = mesh_obj["element"]
    tri = mesh_obj["node"]
    perm = mesh_obj["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_x = np.round(tri_centers[:,0]*100)
    mesh_y = np.round(tri_centers[:,1]*100)
    Perm = np.ones(tri_centers.shape[0])
    for i in range(len(X)):#IMG koordinaten
        for j in range(len(mesh_x)):# tri_centers
            if X[i] == mesh_x[j] and Y[i] == mesh_y[j]:
                Perm[j] = 10
    mesh_obj['perm'] = Perm
    return mesh_obj


def template_all(objct='circle',x=0.5,y=0, phi=0, d=0.2):
    """
    input:  - single Object: 'circle', 'square', 'triangle'
            - x,y
            - phi
            - d
    return: IMG (100x100 pixel picture)
    """
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def draw_circle(x,y,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = (x,y)#*0 für die Rotation
        color = (1,0,0)
        thickness = -1
        IMG = cv2.circle(IMG, center_coordinates, int(d*100), color, thickness)
        return IMG

    def draw_square(x,y,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = (x,y)#*0 für die Rotation
        start_point = (int(center_coordinates[0]-int(d*100)), center_coordinates[1]-int(d*100))
        end_point = (center_coordinates[0]+int(d*100), center_coordinates[1]+int(d*100))
        color = (1, 0, 0)
        thickness = -1
        IMG = cv2.rectangle(IMG, start_point, end_point, color, thickness)
        return IMG
    
    def draw_triangle(x,y,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = (x,y)
        pt1 = (int(center_coordinates[0]), int(center_coordinates[1]-int((d)*100)))
        pt2 = (int(center_coordinates[0]+int((d)*100)), int(center_coordinates[1])+int((d)*100))
        pt3 = (int(center_coordinates[0]-int((d)*100)), int(center_coordinates[1])+int((d)*100))
        tri_edges = np.array( [pt1, pt2, pt3] )
        IMG = cv2.drawContours(IMG, [tri_edges], 0, (1,0,0), -1)
        return IMG
            
    x = int(x*100)+100
    y = y*-1
    y = int(y*100)+100
    angle = phi #Needed for rotation
    phi = math.radians(phi)# Grad in Rad

    if objct == 'circle':
        IMG = draw_circle(x,y,phi,d)
    if objct == 'square':
        IMG = draw_square(x,y,phi,d)
    if objct == 'triangle':
        IMG = draw_triangle(x,y,phi,d)
        
    IMG = rotate_image(IMG,angle)    
    return IMG

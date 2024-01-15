import numpy as np 
import glob
import matplotlib.pyplot as plt

path_files=glob.glob('./ws/curves/*/*.npz')

for p in path_files:
    data=np.load(p)
    dat_name="FPHA" if "fpha" in p else "H2O"
    hand="L" if "left" in p else "R"
    aligned="-RA" if "cent" in p else ""


    fig = plt.figure(figsize=(8 if aligned=="" else 5,5),facecolor='white')
    plt.ylabel('3D PCK'+aligned,fontsize=20, family='serif', weight='bold')
    plt.xlabel('Error threshold/cm',fontsize=20, family='serif',  weight='bold')
    plt.yticks(np.linspace(0,1.0,11),fontsize=16,family='serif', weight='bold')
    plt.xticks(list(range(0,int(data["x"][-1])+1)),fontsize=16,family='serif',  weight='bold')
    plt.title(dat_name+"-"+hand, loc='center',fontsize=20, family='serif',  weight='bold')
    plt.grid(linestyle='-.')


    plt.xlim((0, data["x"][-1]+0.01))
    plt.ylim((0, 1.01))

    plt.plot(data["x"],data["y"],linewidth=3)
    plt.tight_layout()

    plt.savefig('pck_curve.png')
    plt.show()
    plt.close()
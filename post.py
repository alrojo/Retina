import matplotlib.pyplot as plt
import numpy as np
import matplotlib

alex2 = np.load('metadata/dump_alex-20150516-1920520.pkl')
alex2.keys()
for key, value in alex2.iteritems():
    print "type for %s is: %s" % (key, type(value))
alex2['config_name']
for i1, i2 in enumerate(alex2['param_values']):
    print i1, i2.shape

    
for i0, layer in enumerate(alex2['param_values']): 
    if len(layer.shape)<2:
        continue
    filters, depth = layer.shape[0:2] 
    fig, axes = plt.subplots(filters,depth)
    for i1, f in enumerate(layer):
        for i2, mask in enumerate(f): 
            ax = axes[i1,i2]
            ax.matshow(mask,cmap=matplotlib.cm.cool)
    plt.show()

# fig.colorbar(orientation='horizontal')
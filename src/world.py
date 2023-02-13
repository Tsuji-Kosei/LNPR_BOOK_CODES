import matplotlib
import os
if os.name == 'nt':
    print('on Windows')
    matplotlib.use('qtagg')
elif os.name == 'posix':
    print('on Mac or Linux')
    matplotlib.use('MacOSX')
import matplotlib.animation as anm
import matplotlib.pyplot as plt

class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval 
        
    def append(self,obj):  
        self.objects.append(obj)
    
    def draw(self): 
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')             
        ax.set_xlim(-5,5)                  
        ax.set_ylim(-5,5) 
        ax.set_xlabel("X",fontsize=10)                 
        ax.set_ylabel("Y",fontsize=10)                 
        
        elems = []
        
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): 
                self.one_step(i, elems, ax)
        else:
            # self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
            #                          frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()
        
    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)   
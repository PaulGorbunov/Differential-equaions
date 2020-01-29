import re
import math
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from tkinter import ttk
from time import time
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure
from functools import*
from math import *
from multiprocessing import Process
from PIL import Image                                                                                



app_name = "differential equation (IVP)"
back='#66CDAA'
font = "Verdana 10"
sp = " "

my_task = ("0.01","y/x + x*cos(x)","1","pi","4*pi","y","x")
m_ans = "x*sin(x) + x*((1-pi*sin(pi))/pi)" 
err_gr_range = np.arange(0.01,1,0.1)
var_data = []

class Graph:
    def __init__(self):
        self._ind = 111
        
    def create_figure(self,data,info="input parameters",fl=False,xw=9,yh=6):
        try :
            if not fl:
                self.fig.delaxes(self.fig.axes[0])
            self.fig
        except AttributeError:
            self.fig = Figure(figsize=(xw,yh), dpi=100)
        figure = self.fig.add_subplot(self._ind)
        figure.grid(True)
        m = (lambda: "-" if fl else "--")()
        figure.plot(data[0],data[1],m,label = info)
        figure.legend(loc='upper left')
            
    def create_canvas(self,wh=False):
        try:
            self.fig
        except AttributeError:
            if wh:
                self.create_figure([1,1],"input parameters",False,wh[0],wh[1])
            else:
                self.create_figure([1,1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)  
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.canvas.draw()
        self.canvas.flush_events()
        
    def enable_toolbar(self):
        try:            
            toolbarFrame = tkinter.Frame(master=self.frame)
            toolbarFrame.grid(row=1,column=0)
            toolbar = NavigationToolbar2Tk(self.canvas,toolbarFrame)
            toolbar.config(bg=back)
            toolbar._message_label.config(bg=back)
            toolbar.update()
        except AttributeError:
            print("error: crate figure first")
            
    def create_graph(self,**opt):
        self.frame = tkinter.Frame(master=opt["mas"],bg=back)
        self.frame.grid(row=opt["row"],column=opt["col"])
        if "wh" in opt.keys():
            self.create_canvas(opt["wh"])
        else:
            self.create_canvas()
        if(opt['toolbar']):
            self.enable_toolbar()
            
class IVP_app:
    def __init__(self,name,fl):
        self.root = tkinter.Tk()
        self.root.resizable(width=False, height=False)
        self.root.style = ttk.Style()
        self.root.style.theme_use("classic")
        self.root.wm_title(name)
        self.root.configure(bg=back)
        self.main_interf = Graph()
        self.wifl = fl
    
    def create_app(self):
        top_frame = tkinter.Frame(self.root, height=10,bg=back).grid(row=0, columnspan=6)
        objects = Actions(self,row=1,col=1,mas={"main":self.main_interf})
        objects.create_resp("ent","Enter parameters:",1)
        names_inp = ["Grid range:","Equation: dy/dx =","y0:","x0:","X:","Function:","Parameter:"]
        [objects.create_inp(u,names_inp[u],1) for u in range(len(names_inp))]
        objects.create_resp("ans","Ans:"+sp,1)
        objects.create_resp("err","Error:"+sp)
        objects.create_radio("Show error")
        p_b = [(Actions.calc,Functions.euler),(Actions.calc,Functions.imp_euler),(Actions.calc,Functions.runge_kutta),(Actions.var_task,Functions.variant),(Actions._clear,1),(Actions._quit,0)]
        n_b = ["Euler’s method","Improved Euler’s method","Runge-Kutta method","Variant task","Clear graph","Quit"]
        [objects.create_button(n_b[u],p_b[u][0],p_b[u][1]) for u in range(len(n_b))]
        objects.create_space(row=1,col=2,wid=5,mas=self.root)
        self.main_interf.create_graph(toolbar=True,row=1,col=3,mas=self.root)
        objects.create_space(row=1,col=4,wid=5,mas=self.root)
        but_frame = tkinter.Frame(self.root,height=10,bg=back).grid(row=5, columnspan=6)
    
    def create_widg(self):
        self.main_interf2 = Graph()
        top_frame = tkinter.Frame(self.root, height=20,bg=back).grid(row=0, columnspan=6)
        objects = Actions(self,row=0,col=0,mas={"main":self.main_interf,"near":self.main_interf2})
        objects.create_space(row=1,col=0,wid=5,mas=self.root)
        self.main_interf.create_graph(toolbar=True,row=1,col=1,wh=[6,5],mas=self.root)
        objects.create_space(row=1,col=2,wid=5,mas=self.root)
        self.main_interf2.create_graph(toolbar=True,row=1,col=3,wh = [6,5],mas=self.root)
        objects.create_space(row=1,col=4,wid=5,mas=self.root)
        but_frame = tkinter.Frame(self.root,height=20,bg=back).grid(row=5, columnspan=6)
        objects.start_widg()
        
    def start(self):
        self.create_app()
        if self.wifl:
            my_widg = IVP_app(app_name,False)
            my_widg.create_widg()
        tkinter.mainloop()
        
class Actions():
    def __init__(self,interf,**opt):
        self.interf = opt["mas"]
        self.root_interf = interf
        self.entry_d = {}
        self.frame = tkinter.Frame(master=self.root_interf.root,bg=back)
        self.frame.grid(row=opt['row'],column=opt['col'])
        self.row = 0
        self.resp = {}
        self.r = False
        self.r_ch_f = lambda: ("On","#8DB600") if self.r else ("Off","#9C2542")
    
    def create_space(self,pluss=False,**opt):
        space = tkinter.Frame(master=opt["mas"],bg=back)
        space.grid(row=opt["row"],column=opt["col"])
        tkinter.Label(space,text=" "*opt["wid"],bg=back).grid(row=0,column=0)
        if pluss:
            self.row+=1
        
    def create_button(self,name,funct,*param):
        act = partial(funct,self,name,*param,)
        button = tkinter.Button(master=self.frame, text=name, command=act,bg="#FF6347")
        button.grid(row=self.row,column=1)
        self.row+=1
        if not(type(param[-1]) == int and param[-1] == 0):
            self.create_space(True,row=self.row,col=0,wid=1,mas=self.frame)
        
    def create_resp(self,_id,name,sp=0):
        self.resp[_id] = tkinter.Label(self.frame, text=name,bg=back,font =font+" bold")
        self.resp[_id].grid(row=self.row,columnspan=4)
        self.row+=1
        if sp == 0:
            self.create_space(True,row=self.row,col=0,wid=1,mas=self.frame)
    
    def create_inp(self,_id,name,*param):
        tkinter.Label(self.frame, text=name,bg=back,font =font).grid(row=self.row,column=0)
        entry = tkinter.Entry(self.frame)
        entry.bind("<Return>",lambda x: x)
        entry.grid(row=self.row,column=1)
        self.row+=1
        self.entry_d[_id] = entry
        if not(type(param[-1]) == int and param[-1] == 0):
            self.create_space(True,row=self.row,col=0,wid=1,mas=self.frame)
    
    def create_radio(self,name):
        act = partial(Actions.radio,self)
        but = tkinter.Button(master=self.frame, text=name, command=act,bg="#B2BEB5",font="Verdana 8")
        but.grid(row=self.row-1,column=0)
        t = self.r_ch_f()
        self.resp["rad"] =  tkinter.Label(self.frame, text=t[0],bg=t[1],font = font)
        self.resp["rad"].grid(row=self.row-2,columnspan=1)
        
    def radio(self):
        self.r = not self.r
        t = self.r_ch_f()
        self.resp["rad"].configure(text=t[0],bg=t[1])
    
    def calc(self,name,*inp):
        data = Functions.main(inp[0],self.entry_d,True,self.r,name+" (eq: "+self.entry_d[1].get()+" )")
        if (type(data) == int):
            mes = "Error in input"
            if data == 2:
                mes = "Error Zero Division"
            self.resp["ent"].configure(text =mes ,bg = "#FA8072",fg = "#FFFFE0",font = font+" bold")
            self.resp["ans"].configure(text ="Ans:"+sp+mes,bg = "#FA8072",fg = "#FFFFE0",font = font+" bold")
            return 0
        self.resp["ans"].configure(text ="Ans:"+sp+str(round(data[1][-1],4)) ,bg = back,fg = "black",font = font+" bold")
        self.resp["ent"].configure(text = "Enter parameters:",bg = back,fg = "black",font=font+" bold")
        if len(data) > 2:
            self.resp["err"].configure(text = "Error:"+sp+str(round(data[2],4)),bg = back,fg = "black",font=font+" bold")
        else:
            self.resp["err"].configure(text = "Error:"+sp,bg = back,fg = "black",font=font+" bold")
        self.draw_data(data,name,"main",True)
    
    def var_task(self,name,*inp):
        for u in self.entry_d.keys():
            self.entry_d[u].delete(0,tkinter.END)
            self.entry_d[u].insert(0,my_task[u])
        data = Functions.main(inp[0],self.entry_d,False)
        self.resp["ans"].configure(text ="Ans:"+sp+str(round(data[1][-1],4)) ,bg = back,fg = "black",font = font+" bold")
        self.resp["ent"].configure(text = "Enter parameters:",bg = back,fg = "black",font=font+" bold")
        self.resp["err"].configure(text = "Error:"+sp,bg = back,fg = "black",font=font+" bold")
        self.draw_data(data,name,"main")
        
    def _clear(self,name,*inp):
        try :
            os.remove("tmp.png")
        except FileNotFoundError:
            pass
        global var_data
        var_data = []
        for u in self.entry_d.keys():
            self.entry_d[u].delete(0,tkinter.END)
        self.resp["ans"].configure(text ="Ans:"+sp ,bg = back,fg = "black",font = font+" bold")
        self.resp["ent"].configure(text = "Enter parameters:",bg = back,fg = "black",font=font+" bold")
        self.resp["err"].configure(text = "Error:"+sp,bg = back,fg = "black",font=font+" bold")        
        self.draw_data([1,1],name,"main")

    def draw_data(self,data,name,mas,fl=False):
        self.interf[mas].create_figure(data,name+" (eq: "+self.entry_d[1].get()+" grid: "+self.entry_d[0].get()+")",fl)
        self.interf[mas].canvas.draw()
        self.interf[mas].canvas.flush_events()
        
    def _quit(self,*inp):
        try :
            os.remove("tmp.png")
        except FileNotFoundError:
            pass
        self.root_interf.root.quit()     
        self.root_interf.root.destroy()  

    def start_widg(self):
        pass
            
class Functions:
    @staticmethod
    def main(funct,inp_d,fl,e_sh=False,name=""):
        try:
            grid = eval(inp_d[0].get())
            y0 = eval(inp_d[2].get())
            x0 = eval(inp_d[3].get())
            X = eval(inp_d[4].get())
            f_n = inp_d[5].get()
            p_n = inp_d[6].get()
        except NameError:
            return 1
        except SyntaxError:
            return 1
        equa,par = Functions.get_equation((lambda: m_ans if not fl else inp_d[1].get())())
        if (len(equa)== 0 or (not f_n in par) or (len(par)>1 and not p_n in par)) and fl:
            return 1
        try :
            equa = Functions.equation(equa,par)
            calc = partial(Functions.calc,f_n,p_n,equa)
            ans = funct(calc,x0,y0,X,grid)
            if e_sh:
                Process(target = Functions.error,args=(calc,x0,y0,X,grid,funct,name)).start()
            if len(var_data) == 2 :
                ans.append(abs(var_data[1][-1]-ans[1][-1]))
            return ans
        except ZeroDivisionError:
            return 2
        except SyntaxError:
            return 1
        except KeyError:
            return 1
    @staticmethod
    def get_equation(eq):
        reserved = [m for m in dir(math)]
        par = re.findall("[a-zA-Z]+",eq)
        par = [x for x in par if not x in reserved]
        par = list(dict.fromkeys(par))
        return eq,par
    
    @staticmethod
    def equation(eq,par):
        d = {}
        for u in par:
            d[u] = 0
            eq = eq.replace(u,"d['"+u+"']")
        return eq

    @staticmethod
    def calc(fn,pn,eq,x,y):
        d= {pn:x,fn:y}
        return eval(eq)

    @staticmethod
    def euler(calc,x0,y0,X,grid):
        xs = np.arange(x0,X,grid)
        ys = [y0]
        _der = [calc(x0,y0)]
        def step(x):
            ys.append(ys[-1]+_der[-1]*grid)
            _der.append(calc(x,ys[-1]))
        [step(x) for x in xs[1:]]
        return [xs,ys]
    
    @staticmethod
    def imp_euler(calc,x0,y0,X,grid):
        xs = np.arange(x0,X,grid)
        ys = [y0]
        _der1 = [calc(x0,y0)]
        _der2 = [calc(x0+grid,ys[-1]+grid*_der1[-1])]
        get_y = lambda: (_der1[-1] + _der2[-1])/2
        def step(x):
            ys.append(ys[-1] + grid * get_y())
            _der1.append(calc(x,ys[-1]))
            _der2.append(calc(x+grid,ys[-1]+grid*_der1[-1]))
        [step(x) for x in xs[1:]]
        return [xs,ys]
    
    @staticmethod
    def runge_kutta(calc,x0,y0,X,grid):
        xs = np.arange(x0,X,grid)
        ys = [y0]
        f = lambda g,s : f(g*10,s+1) if g < 1 else (g,s)
        dx = 10**(f(grid,0)[1]+1.5)
        integr = lambda x,y: sum([calc(_x,y)/dx for _x in np.arange(x*dx,(x+grid)*dx,1)/dx])        
        def step(x):
            ys.append(ys[-1] + integr(x,ys[-1]))
        [step(x) for x in xs[1:]]
        return [xs,ys]
    
    @staticmethod
    def variant(calc,x0,y0,X,grid):
        global var_data
        xs = np.arange(x0,X,grid)
        ys = [y0]
        def step(x,y=0):
            ys.append(calc(x,y))
        [step(x) for x in xs[1:]]
        var_data = [np.round(np.array(xs),3),np.round(np.array(ys),3)]
        return [xs,ys]
        
    @staticmethod
    def error(calc,x0,y0,X,grid,funct,name):
        if len(var_data) != 2 :
            print("error")
            return 1
        xs = err_gr_range[::-1]
        def count(d_lis):
            a_xs = [np.round(np.array(x[0]),3) for x in d_lis]
            a_ys = [y[1] for y in d_lis]
            i = [0]
            def _i(fl):
                if fl == 0:
                    return i[-1]-1
                elif fl == 1:
                    i[-1] +=1
                    return True
                else:
                    i[-1] = 0
                    return True
            inds = [[(_i(0),q[0]) for q in w if len(q)>0 and _i(1)] for w in np.transpose([[np.where( var_data[0] == u)[0] for u in q] for q in a_xs]) if _i(2)]
            return [sum([abs((a_ys[rlis][l[0]]-var_data[1][l[1]]))/len(inds[rlis]) for l in inds[rlis]]) for rlis in range(len(inds))]
        ys = count([funct(calc,x0,y0,X,gr) for gr in err_gr_range])
        plt.plot(xs, ys,label = "final error ~ "+str(round(ys[0],4)))
        plt.title("Error graph: "+name)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.savefig('tmp.png')
        img = Image.open('tmp.png')
        img.show() 
    
if __name__ == "__main__":
    my_app = IVP_app(app_name,True)
    my_app.start()

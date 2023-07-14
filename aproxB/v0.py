from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

#Mallat:
lbase = 1 #cm
laltura = 3 #cm
mesh = RectangleMesh(Point(0, 0), Point(lbase, laltura), 6*lbase, 3*laltura)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)


# Condicions de contorn:
base  = 'near(x[1], 0)'
tapa = 'near(x[1], 3)'
parets   = 'near(x[0], 0) || near(x[0], 1)'

u_parets  = DirichletBC(V, Constant((0, 0)), parets)
u_tapa  = DirichletBC(V, Constant((0, 0)), tapa)
u_base  = DirichletBC(V, Constant((0, 0)), base)
p_tapa  = DirichletBC(Q, Constant(101300), tapa)
p_base = DirichletBC(Q, Constant(101300), base)
c_parets  = DirichletBC(Q, Constant(0), parets)
c_tapa = DirichletBC(Q, Constant(0), tapa)
c_base = DirichletBC(Q, Constant(0), base)

bcu = [u_parets]
bcp = [p_tapa, p_base]
#bcc = [c_base]
bcc = [c_parets, c_tapa, c_base]

# Funcions trial/test:
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
c_trial=TrialFunction(Q)
c_test=TestFunction(Q)

# Funcions solució t-1:
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)
c_n=Function(Q)
c_=Function(Q)

# Interval temporal:
T = 30000 #s
num_steps = 60000
dt = T / num_steps 

# Constants:
U   = 0.5*(u_n + u) #cm/s
n   = FacetNormal(mesh)
k   = Constant(dt) #s
mu  = Constant(8.52*10**(-4)) #kg/ms
rho = Constant(1) #m 
m=Constant(5*10**(-20)) #kg 

muo=Constant(4*np.pi*10**(-7)) #N/A^2
Br=Constant(1.45) #T????
a=Constant(0.7) #radi iman (cm)
h=Constant(1.5) #alçada iman(cm)

altura=Constant(laltura)
kb=Constant(1.38*10**(-23))
temp=Constant(300)

g=Constant(9.81) #m/s^2
Fg=m*g

D=Constant(1)
patm=Constant(101300)
rho=Constant(1)
rh=Constant(15*10**(-9)) #m 


# Força magnètica:
gradB=Expression('0.5*%e*0.7*0.7*(pow(pow(x[1]+%e,2)+0.7*0.7,-1.5)-pow(pow(x[1],2)+0.7*0.7,-1.5))'%(Br,h),degree=2)
derivada_gradB=Expression('0.5*%e*0.7*0.7*(3*(3+x[1])*pow(pow(x[1]+%e,2)+0.7*0.7,-2.5)-3*x[1]*pow(pow(x[1],2)+0.7*0.7,-2.5))'%(Br,h),degree=2)
#FALTA FER LA DERIVADA

#gradB=Constant(-1)
c_n.assign(Constant(pow(10,-5))) #REVISAR VALOR!!
p_n.assign(patm)
u_n.assign(Constant([0,0]) )
#M=Constant([0,0]) #Am^2/kg
M=Constant([0,42.7]) #Am^2/kg
f=gradB*c_n*M
v_part=(m*M*g)/(6*np.pi*mu*rh)*gradB*100
div_vpart=(m*42.7*g)/(6*np.pi*mu*rh)*derivada_gradB*10000
eixY=Constant([0,1]) 

#Definir Navier-Stokes:

# Strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Eq 26 (pas 1)
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(gradB*c_n*M, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Eq 27 (pas 2)
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Eq 28 (pas 3)
a3 = dot(u, v)/k*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Concentració
F4 = ((c_trial - c_n)/k*c_test - dot(u_, grad(c_trial))*c_test\
     - (dot(grad(c_trial),v_part)+c_trial*div_vpart)*c_test) * dx
#F4 = ((c_trial - c_n)/k*c_test - dot(u_, grad(c_trial))*c_test\
#     - (dot(grad(c_trial),v_part)+c_trial*div(v_part))*c_test) * dx
#F4 = ((c_trial - c_n)/k*c_test - dot(u_, grad(c_trial))*c_test\
#     - dot(grad(c_trial)*c_test, v_part) + D*dot(grad(c_trial),grad(c_test))) * dx
a4 = lhs(F4)
L4 = rhs(F4)

# Assemble
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)

# Aplicar condicions de contorn:
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
[bc.apply(A3) for bc in bcu]
[bc.apply(A4) for bc in bcc]
# Bucle temporal:
t = 0
a=0
concentracions_max=[pow(10,-5)]
temps=[t]
for n in range(num_steps):
    a+=1
    t += dt

    # Trobar velocitat intermitja
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Corretgir pressió
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Corretgir velocitat
    b3 = assemble(L3)
    [bc.apply(b3) for bc in bcu]
    solve(A3, u_.vector(), b3)
    

    # Trobar concnetració
    b4 = assemble(L4)
    [bc.apply(b4) for bc in bcc]
    solve(A4, c_.vector(), b4)

    # Plot
    '''
    cmax=np.array(c_.vector()).max()
    concentracions_max.append(cmax)
    temps.append(t)
    plot(c_)
    c=plot(c_)
    plt.colorbar(c)
    plt.draw()
    plt.pause(0.01)
    plt.clf()
    '''
    #Guardar en document
    
    if a%100==0 or a<30:
        #print(t)
        plot (u_)
        color_u=plot(u_)
        plt.colorbar(color_u)
        plt.title ( 'Velocitats t=%.2fs' % (t) )
        filename = ( 'v_t%d.png' % (t) )
        plt.savefig ( filename )
        #print ( '  Guardat document "%s"' % ( filename ) )
        plt.close ( )
    if a%100==0 or a<30:
        #print(t)
        plot (c_)
        color_c=plot(c_)
        plt.colorbar(color_c)
        plt.title ( 'Concentracions t=%.2fs' % (t) )
        filename = ( 'c_t%d.png' % (t) )
        plt.savefig ( filename )
        print ( '  Guardat document "%s"' % ( filename ) )
        plt.close ( )
    if a==1:
        plot (p_)
        color_p=plot(p_)
        plt.colorbar(color_p)
        plt.title ( 'Pressions t=%.2fs' % (t) )
        filename = ( 'p_t%d.png' % (t) )
        plt.savefig ( filename )
        print ( '  Guardat document "%s"' % ( filename ) )
        plt.close ( )
    
    # Comprovar si s'estabilitza
    #print('max u:', np.array(u_.vector()).max())

    # Guardar les solucions
    u_n.assign(u_)
    p_n.assign(p_)
    c_n.assign(c_)




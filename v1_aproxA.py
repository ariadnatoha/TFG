from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

#Mallat:
lbase = 1
laltura = 3
mesh = RectangleMesh(Point(0, 0), Point(lbase, laltura), 6*lbase, 3*laltura)
#mesh = RectangleMesh(Point(0, 0), Point(lbase, laltura), 6, 8)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Condicions de contorn:
base  = 'near(x[1], 0)'
tapa = 'near(x[1], 3)'
parets   = 'near(x[0], 0) || near(x[0], 1)'

u_parets  = DirichletBC(V, Constant((0, 0)), parets)
u_tapa  = DirichletBC(V, Constant((0, 0)), tapa)
u_base  = DirichletBC(V, Constant((0, 0)), base)
p_tapa  = DirichletBC(Q, Constant(1), tapa)
p_base = DirichletBC(Q, Constant(1), base)

bcu = [u_parets, u_base, u_tapa]
bcp = [p_tapa, p_base]

# Funcions trial/test:
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Funcions solució t-1:
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

c_n=Function(Q)

# Interval temporal:
T = 10.0
num_steps = 100
dt = T / num_steps 

# Constants:
U   = 0.5*(u_n + u)
n   = FacetNormal(mesh)
k   = Constant(dt)
mu  = Constant(8.52*10**(-4)) #kg/ms
rho = Constant(1) #m 
m=Constant(5*10**(-20)) #kg 

muo=Constant(4*np.pi*10**(-7)) #N/A^2
Br=Constant(1.45) #T????
a=Constant(0.7) #radi iman (cm)
h=Constant(1.5) #alçada iman(m)

altura=Constant(laltura)
kb=Constant(1.38*10**(-23))
temp=Constant(300)

g=Constant(9.81) #m/s^2
Fg=m*g

D=Constant(1)
patm=Constant(1)
rho=Constant(1)
po=patm+3*rho*g

# Força magnètica:
gradB=Expression('0.5*%e*0.7*0.7*(pow(pow(x[1]+%e,2)+0.7*0.7,-2.5)-pow(pow(x[1],2)+0.7*0.7,-2.5))'%(Br,h),degree=2)
#gradB=Constant(-1)
c_n.assign(Constant(pow(10,-5)))
p_n.assign(patm)
#M=Constant([0,0]) #Am^2/kg
M=Constant([0,42.7]) #Am^2/kg
f=gradB*c_n*M


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
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Eq 27 (pas 2)
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Eq 28 (pas 3)
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Aplicar condicions de contorn:
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]


# Bucle temporal:
t = 0
a=0
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
    solve(A3, u_.vector(), b3)

    # Plot
    plot(u_)
    c=plot(u_)
    plt.colorbar(c)
    plt.draw()
    plt.pause(0.1)
    plt.clf()

    #Guardar en document
    if a%10==0:
        print(t)
        plot (u_)
        c=plot(u_)
        plt.colorbar(c)
        plt.title ( 'Velocitats t=%.2fs' % (t) )
        filename = ( 'velocitats_t%d.png' % (t) )
        plt.savefig ( filename )
        print ( '  Guardat document "%s"' % ( filename ) )
        plt.close ( )
    if a==1:
        plot (p_)
        c=plot(p_)
        plt.colorbar(c)
        plt.title ( 'Pressions t=%.2fs' % (t) )
        filename = ( 'pressions_t%d.png' % (t) )
        plt.savefig ( filename )
        print ( '  Guardat document "%s"' % ( filename ) )
        plt.close ( )
    
    # Comprovar si s'estabilitza
    print('max u:', np.array(u_.vector()).max())

    # Guardar les solucions
    u_n.assign(u_)
    p_n.assign(p_)




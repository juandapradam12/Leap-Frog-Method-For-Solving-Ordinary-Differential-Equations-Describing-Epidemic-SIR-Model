#### Modelo SIR

import numpy as np 
import matplotlib.pylab as plt

# Puntos 

h = 0.001
min_t = 0.0
max_t = 17.0

n_puntos=int((max_t-min_t)/h)

t = np.zeros(n_puntos) 

# Tiempo 

t[0] = min_t
t[1]= min_t + h

#### Caso 1 

# Constantes 1
betha_1 = 0.0022
gamma_1 = 0.45 

# Arrays 

S_1 = np.zeros(n_puntos)
I_1 = np.zeros(n_puntos)
R_1 = np.zeros(n_puntos)

def S_prima_1(S, I):
	return -betha_1*I*S

def I_prima_1(S, I):
	return betha_1*I*S-gamma_1*I

def R_prima_1(I):
	return gamma_1*I

# Condiciones iniciales 

N = 771
I_1[0] = 1.0
S_1[0] = 770
R_1[0] = 0.0

# Primer paso 

S_1[1] = S_1[0] + h*S_prima_1(S_1[0], I_1[0])
I_1[1] = I_1[0] + h*I_prima_1(S_1[0], I_1[0])
R_1[1] = R_1[0] + h*R_prima_1(I_1[0])

# Integracion 

for i in range(2,n_puntos):
	t[i] = t[i-1] + h
	S_1[i] = S_1[i-2] + 2*h*S_prima_1(S_1[i-1], I_1[i-1])
	I_1[i] = I_1[i-2] + 2*h*I_prima_1(S_1[i-1], I_1[i-1])
	R_1[i] = R_1[i-2] + 2*h*R_prima_1(I_1[i-1])

#### Caso 2

# Constantes 2 

betha_2 = 0.001
gamma_2 = 0.2

# Arrays 

S_2 = np.zeros(n_puntos)
I_2 = np.zeros(n_puntos)
R_2 = np.zeros(n_puntos)

def S_prima_2(S, I):
	return -betha_2*I*S

def I_prima_2(S, I):
	return betha_2*I*S-gamma_2*I

def R_prima_2(I):
	return gamma_2*I
# Condiciones iniciales 

N = 771
I_2[0] = 1.0
S_2[0] = 770
R_2[0] = 0.0

# Primer paso 

S_2[1] = S_2[0] + h*S_prima_2(S_2[0], I_2[0])
I_2[1] = I_2[0] + h*I_prima_2(S_2[0], I_2[0])
R_2[1] = R_2[0] + h*R_prima_2(I_2[0])

# Integracion 

for i in range(2,n_puntos):
	t[i] = t[i-1] + h
	S_2[i] = S_2[i-2] + 2*h*S_prima_2(S_2[i-1], I_2[i-1])
	I_2[i] = I_2[i-2] + 2*h*I_prima_2(S_2[i-1], I_2[i-1])
	R_2[i] = R_2[i-2] + 2*h*R_prima_2(I_2[i-1])

#### Grafica con resultados de ambos casos .pdf 

soluciones = plt.figure() 

sln_caso_1 = soluciones.add_subplot(211)
sln_caso_2 = soluciones.add_subplot(212)

sln_caso_i = [sln_caso_1, sln_caso_2]

SIR = ["Susceptibilidad", "Infectados", "Recuperados"]

SIR_1 = [S_1, I_1, R_1]

for sir_1, sir in zip(SIR_1, SIR):
	sln_caso_1.plot(t, sir_1, label = sir)
		
SIR_2 = [S_2, I_2, R_2]

for sir_2, sir in zip(SIR_2, SIR):
	sln_caso_2.plot(t, sir_2, label = sir)

ubicacion = ["upper right", "upper left"]

for sln, ub in zip(sln_caso_i, ubicacion):
	sln.legend(loc = ub)
	
plt.savefig('SIR.pdf')
#plt.show()


#### Maximo de infectados en dias

Max_I_caso_1 = np.argmax(I_1)
Max_I_caso_2 = np.argmax(I_2)

dias_caso_1 = t[Max_I_caso_1]
dias_caso_2 = t[Max_I_caso_2]


print("Caso 1, tiempo I_max(dias): {} y Caso 2, tiempo I_max(dias): {}." .format(dias_caso_1, dias_caso_2))





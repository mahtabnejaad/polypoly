#!/bin/bash

# exeName

# input x-v-data file name
input="1"

# output base filename
basename="0.1"

# Dimension of the simulation
L0=50
L1=50
L2=50

# density of mpcd particles:
d=10

#number of rings:
n=1
#number of monomer in each ring 
m=100

# shear rate
s=0.0

#md time step:
hmd=0.00400

#mpcd time step:
hmpcd=0.02

# output intervall (swapsize)
Tout=1

# simulate time 
Tsim=200

#start time: 
t=0

#topology 0 for ring 1 for poltycatenane 2 for linked ring , 3 for ... and 4 for one active particle in a fluid.
topology=1

#Activity on or off (0 o1 1):
Activity=1

#Random flag :
random_flag=0

# kind of Boundary condition : 1 is for periodic BC, 2 is for no-slip BC, and 3 is for combined BC(not developed yet).
BC=2

# peclet number
Pe=1

# K(FENE) adjusting with desired pL
K_bend=10.5

./main.run  $input $basename $L0 $L1 $L2 $d $n $m $s $hmd $hmpcd $Tout $Tsim $t $topology $Activity $random_flag $BC $Pe $K_bend





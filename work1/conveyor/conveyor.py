from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import math
import csv

Pi = math.pi

robotNum = 'GP8'
client = RemoteAPIClient()
sim = client.require('sim')

executedMovId = 'notReady'
targetArm = '/yaskawa'
stringSignalName = targetArm + '_executedMovId'
sim.setStepping(True)

sim.startSimulation()

cup=sim.getObject('./conveyorSystem/Cup')

t = []
# linear
cupPx = []
cupPy = []
cupPz = []
cupVx = []
cupVy = []
cupVz = []
cupAx = []
cupAy = []
cupAz = []
# angular
cupP_alpha = []
cupP_beta = []
cupP_gamma = []
cupV_alpha = []
cupV_beta = []
cupV_gamma = []

i = 0
dt = sim.getSimulationTimeStep()
targetTime = 33.5
step = int(targetTime / dt)

while not sim.getSimulationStopping():
    print('t %d: %.2f sec' % (i, sim.getSimulationTime()))
    cupPos = sim.getObjectPosition(cup, sim.handle_world)
    alpha, beta, gamma = sim.getObjectOrientation(cup, sim.handle_world)
    cupL_Vel, cupA_Vel = sim.getObjectVelocity(cup, sim.handle_world)
    # linear
    print('Cup linear')
    print('Cup pos %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupPos[0], cupPos[1], cupPos[2]))
    print('Cup vel %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupL_Vel[0], cupL_Vel[1], cupL_Vel[2]))
    # angular
    print('Cup angular')
    print('Cup angular pos %d: x=%.3f, y=%.3f, z=%.3f ' % (i, alpha, beta, gamma))
    print('Cup angular vel %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupA_Vel[0], cupA_Vel[1], cupA_Vel[2]))
    
    # get data to list
    t.append(sim.getSimulationTime())
    # linear
    cupPx.append(cupPos[0])
    cupPy.append(cupPos[1])
    cupPz.append(cupPos[2])
    cupVx.append(cupL_Vel[0])
    cupVy.append(cupL_Vel[1])
    cupVz.append(cupL_Vel[2])
    # angular
    cupP_alpha.append(alpha)
    cupP_beta.append(beta)
    cupP_gamma.append(gamma)
    cupV_alpha.append(cupA_Vel[0])
    cupV_beta.append(cupA_Vel[1])
    cupV_gamma.append(cupA_Vel[2])
    
    if i==step:
        break
    i = i + 1
    sim.step()
    
print(dt)  

def plot(t, cupPx, cupPy, cupPz, cupVx, cupVy, cupVz, cupP_alpha, cupP_beta, cupP_gamma, cupV_alpha, cupV_beta, cupV_gamma):
    # plot -------------------------------------------------
    plt.figure # plot linear position

    plt.subplot(2,3,1)
    plt.title('t and linear position x')
    plt.xlabel('t(s)')
    plt.ylabel('m')
    plt.plot(t,cupPx, '--r', label='cup x')
    plt.legend(title='where:')

    plt.subplot(2,3,2)
    plt.title('t and linear position y')
    plt.xlabel('t(s)')
    plt.ylabel('m')
    plt.legend()
    plt.plot(t,cupPy, '--r', label='cup y')
    plt.legend(title='where:')

    plt.subplot(2,3,3)
    plt.title('t and linear position z')
    plt.xlabel('t(s)')
    plt.ylabel('m')
    plt.legend()
    plt.plot(t,cupPz, '--r', label='cup z')
    plt.legend(title='where:')

    plt.subplot(2,3,4)
    plt.title('t and linear velocity x')
    plt.xlabel('t(s)')
    plt.ylabel('m/s')
    plt.legend()
    plt.plot(t,cupVx, '--r', label='cup Vx')
    plt.legend(title='where:')

    plt.subplot(2,3,5)
    plt.title('t and linear velocity y')
    plt.xlabel('t(s)')
    plt.ylabel('m/s')
    plt.legend()
    plt.plot(t,cupVy, '--r', label='cup Vy')
    plt.legend(title='where:')

    plt.subplot(2,3,6)
    plt.title('t and linear velocity z')
    plt.xlabel('t(s)')
    plt.ylabel('m/s')
    plt.legend()
    plt.plot(t,cupVz, '--r', label='cup Vz')
    plt.legend(title='where:')

    plt.show()

    plt.figure # plot angular position

    plt.subplot(2,3,1)
    plt.title('t and angular position x of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad')
    plt.legend()
    plt.plot(t,cupP_alpha, '--r')

    plt.subplot(2,3,2)
    plt.title('t and angular position y of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad')
    plt.legend()
    plt.plot(t,cupP_beta, '--r')

    plt.subplot(2,3,3)
    plt.title('t and angular position z of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad')
    plt.legend()
    plt.plot(t,cupP_gamma, '--r')

    plt.subplot(2,3,4)
    plt.title('t and angular velocity x of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad/s')
    plt.legend()
    plt.plot(t,cupV_alpha, '--r')

    plt.subplot(2,3,5)
    plt.title('t and angular velocity y of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad/s')
    plt.legend()
    plt.plot(t,cupV_beta, '--r')

    plt.subplot(2,3,6)
    plt.title('t and angular velocity z of the cup')
    plt.xlabel('t(s)')
    plt.ylabel('rad/s')
    plt.legend()
    plt.plot(t,cupV_gamma, '--r')

    plt.show()

    fig = plt.figure() # 3D line plot
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    ax.set_aspect('auto')
    ax.plot3D(cupPx, cupPy, cupPz, 'green')
    ax.set_title('3D line plot the cup (x,y,z)')

    plt.show()



# Create CSV file
with open('conveyor/cup_trajectory_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(['Time(s)', 'X(m)', 'Y(m)', 'Z(m)', 'Vx(m/s)', 'Vy(m/s)', 'Vz(m/s)', 'Alpha(rad)', 'Beta(rad)', 'Gamma(rad)', 'V_alpha(rad/s)', 'V_beta(rad/s)', 'V_gamma(rad/s)'])
    
    # Write data rows
    for i in range(len(t)):
        writer.writerow([round(t[i], 6), round(cupPx[i], 6), round(cupPy[i], 6), round(cupPz[i], 6), round(cupVx[i], 6), round(cupVy[i], 6), round(cupVz[i], 6), round(cupP_alpha[i], 6), round(cupP_beta[i], 6), round(cupP_gamma[i], 6), round(cupV_alpha[i], 6), round(cupV_beta[i], 6), round(cupV_gamma[i], 6)])

print('Data exported to cup_trajectory_data.csv')
plot(t, cupPx, cupPy, cupPz, cupVx, cupVy, cupVz, cupP_alpha, cupP_beta, cupP_gamma, cupV_alpha, cupV_beta, cupV_gamma)
# stop program -----------------------------------------
sim.setStepping(False)
sim.stopSimulation()
print('Program ended')

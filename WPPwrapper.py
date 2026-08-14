import mink, mujoco
import numpy as np
from loop_rate_limiters import RateLimiter
import mujoco.viewer, math
from ik_mink import unitree_g1_ik, bxi_elf3_ik


class humanoid_Function:
    # Walk, Pick Place visualization
    states = ["idle",
              "walking", # walking
              "hand-moving"]
    
    yaw_omega = 0.8 # rad/s
    walk_speed = 0.4 # m/s
    
    def _motion_init(self):
        self.default_right = False
        omega_softcap = 0.4 # sped up time/animation play faster if exceeds softcap
        speed_softcap = 0.2 # sped up time/animation play faster if exceeds softcap
        self.yaw_tol = 1e-2
        self.loc_tol = 1e-2

        # motion init
        self._motion = {
            "com_target": [[0,mink.SE3.identity()]],
        }
        ref,current_state = self.ik.get_current_target()
        for key in ["left_palm_target","right_palm_target","left_foot_target","right_foot_target","pelvis_target","chest_target"]:
            self._motion[key] = [[0,current_state[key]]]
        #print(self._motion["right_palm_target"])
        self.t_horizon = 0
        self.leg_states_list = ["standing",# both feet on the ground, idle and ready
                            "moving_right", # left foot on the ground, moving right leg in the air
                            "moving_left"] # right foot on the ground,  moving left leg in the air
        self.leg_state = "standing"
        self.last_speed = 0
        self.last_omega = 0
        self.expected_com = np.array((self.pos_x,self.pos_y,self.yaw)) # for checking early finish

        # gait parameters
        self.above_g_height = 0.05 # lift foot 5 cm above ground
        self.back_leg_tilt = 0.2 # 0.2 rad during lifting
        self.front_leg_tilt = -0.1 # -0.1 rad during landing
        self.lift_ratio = 0.24 # 24% of one step is back leg lifting
        self.land_ratio = 0.24 # 24% of one step is front leg landing
        self.walking_com_height = 0.6 # com 60 cm above ground during walking
        self.standing_foot_y = 0.12 # default standing leg is 12 cm to the side of com
        self.gait_tol = 1e-3 # gait para tolerance

        # walk para
        self.walk_dilation = 1.0
        self.walk_step = self.walk_speed
        # sped up animation if needed
        if self.walk_step > speed_softcap:
            self.walk_step = speed_softcap
            self.walk_dilation = speed_softcap/self.walk_speed
        # turn para
        self.turn_dilation = 1.0
        self.turn_step = self.yaw_omega
        # sped up animation if needed
        if self.turn_step > omega_softcap:
            self.turn_step = omega_softcap
            self.turn_dilation = omega_softcap/self.yaw_omega
        


    def __init__(self,ik_model,init_frame = "init_frame",init_loc=(0,0,0),dt = 0.005):
        self.ik = ik_model
        self.model = self.ik.model
        self.configuration = self.ik.configuration
        self.pos_x,self.pos_y,self.yaw = init_loc
        self._motion_init()
        self.state = "idle"
        self.dt = dt
        self.t = 0
        self.base_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw),np.array([self.pos_x,self.pos_y,self.walking_com_height]))
        self.ik.base_ref = self.base_ref
        self._motion["left_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                    np.array([0,self.standing_foot_y,-self.walking_com_height]))]]
        self._motion["right_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                    np.array([0,-self.standing_foot_y,-self.walking_com_height]))]]
        d_target = self.dance_interpret(self._motion,self.t)
        self.ik.ik(self.base_ref,d_target)
        self.arm_moving_flag =False

    def is_near(self,posa,posb):
        return np.allclose(posa, posb, rtol=1e-03, atol=1e-03)
    
    def delta_yaw(self,current_yaw,desired_yaw):
        result = desired_yaw-current_yaw
        if result > math.pi:
            result -= math.pi*2
        elif result < -math.pi:
            result += math.pi*2
        return result

    # assuming arm will stay the same, only plan leg motion
    def walk_to(self,target=(0,0,0)):
        # time update
        self.t += self.dt
        self.last_target = target
        if self.t >= self.t_horizon:
            self.t = 0
            self.distance_left = math.sqrt((target[1]-self.pos_y)**2+(target[0]-self.pos_x)**2)
            # check early final step
            if np.allclose(self.expected_com,np.array(target),rtol=self.loc_tol,atol=self.loc_tol):
                # on spot
                if self.leg_state == "standing":
                    self.t_horizon = 0
                    self._motion["left_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                                np.array([0,self.standing_foot_y,-self.walking_com_height]))]]
                    self._motion["right_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                                np.array([0,-self.standing_foot_y,-self.walking_com_height]))]]
                    return True
                else:
                    # needs to correct standing pose
                    self.schedule_finish()
            # replan next step
            if self.distance_left<self.loc_tol:
                d_yaw = self.delta_yaw(self.yaw,target[2])
                if -self.yaw_tol<d_yaw<self.yaw_tol:
                    # on spot
                    if self.leg_state == "standing":
                        self.t_horizon = 0
                        self._motion["left_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                                    np.array([0,self.standing_foot_y,-self.walking_com_height]))]]
                        self._motion["right_foot_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),
                                                                                                    np.array([0,-self.standing_foot_y,-self.walking_com_height]))]]
                        return True
                    else:
                        # needs to correct standing pose
                        self.schedule_finish()
                else:
                    # needs to turn
                    self.schedule_step(0,0,d_yaw)
            else:
                # walk a step
                self.yaw_target = math.atan2(target[1]-self.pos_y,target[0]-self.pos_x)
                d_yaw_b = self.delta_yaw(self.yaw,self.yaw_target)
                d_yaw_a = self.delta_yaw(self.yaw_target,target[2])
                self.schedule_step(d_yaw_b,self.distance_left,d_yaw_a)
        # interpret and move robot
        d_target = self.dance_interpret(self._motion,self.t)
        # print(d_target["com_target"])
        self.pos_x = d_target["com_target"].wxyz_xyz[4]
        self.pos_y = d_target["com_target"].wxyz_xyz[5]
        self.yaw = d_target["com_target"].rotation().compute_yaw_radians()
        # update with inverse kinematics
        self.ik.ik(d_target["com_target"],d_target)
        return False


    # the moving leg, preivous yaw, previous x with yaw, end yaw, end x without yaw
    def move_leg(self,pyaw,px_wy,eyaw,ex,moving_left):
        if moving_left:
            lr_factor = 1 # moving left leg
        else:
            lr_factor = -1 # moving right leg
        turn_t = abs(eyaw-pyaw)/self.turn_step
        px = px_wy + lr_factor*self.standing_foot_y*math.sin(pyaw) # without yaw
        ex_wy = ex - lr_factor*self.standing_foot_y*math.sin(eyaw) # with yaw
        walk_t = (ex-px)/self.walk_step
        delta_t = max(turn_t+walk_t,self.lift_ratio+self.land_ratio)
        # initial & final status
        m_previous = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,pyaw),
                                                           np.array([px_wy,lr_factor*self.standing_foot_y*math.cos(pyaw),-self.walking_com_height]))
        m_end = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,eyaw),
                                                           np.array([ex_wy,lr_factor*self.standing_foot_y*math.cos(eyaw),-self.walking_com_height]))
        tilt_factor = min(1,(ex_wy-px_wy)/self.walk_step)
        liftyaw = (eyaw-pyaw)/delta_t*self.lift_ratio+pyaw
        liftx = (ex_wy-px_wy)/delta_t*self.lift_ratio+px_wy
        m_lift = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,tilt_factor*self.back_leg_tilt,liftyaw),
                                                           np.array([liftx,lr_factor*self.standing_foot_y*math.cos(liftyaw),-self.walking_com_height+self.above_g_height]))
        aboveyaw = (eyaw-pyaw)/delta_t*(delta_t-self.land_ratio)+pyaw
        abovex = (ex_wy-px_wy)/delta_t*(delta_t-self.land_ratio)+px_wy
        m_above = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,tilt_factor*self.front_leg_tilt,aboveyaw),
                                                           np.array([abovex,lr_factor*self.standing_foot_y*math.cos(aboveyaw),-self.walking_com_height+self.above_g_height]))
        leg_motion = []
        leg_motion.append([0,m_previous])
        leg_motion.append([self.turn_dilation*self.lift_ratio,m_lift])
        leg_motion.append([self.turn_dilation*(delta_t-self.land_ratio),m_above])
        leg_motion.append([self.turn_dilation*delta_t,m_end])
        return self.turn_dilation*delta_t, leg_motion

    # the expected yaw before forward, forward distance and yaw_after for com
    # the current left x with yaw, current left yaw
    def right_move_one(self,yaw_before,distance,yaw_after,c_left_x_wy,c_left_yaw):
        this_yb,this_d,this_ya = 0,0,0
        right_e_x,right_e_y = 0,0
        if yaw_before >= 0: # turning anti clockwise first
            this_yb = self.turn_step/2+c_left_yaw+self.yaw_tol # maximum yaw supported
            if this_yb < yaw_before: # still full turning 
                right_e_y = min(yaw_before-this_yb,self.turn_step/2)
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            this_yb = yaw_before
            # forward moving
            this_d = c_left_x_wy+self.walk_step/2 # maximum step forward supported
            if this_d + self.walk_step/2 < distance: # full speed forwarding
                right_e_x =self.walk_step/2
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            this_d = min(this_d,distance)
            right_e_x = distance-this_d # small half step
            # turn at the end
            if yaw_after >= 0: # same direction turning
                this_ya = self.turn_step/2+c_left_yaw - this_yb+self.yaw_tol # maximum yaw supported
                if this_ya < yaw_after: # maximum turning
                    right_e_y = self.turn_step/2
                    if yaw_after-this_ya < self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        right_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                    return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                # final landing
                this_ya = yaw_after
                self.expected_com = np.array(self.last_target)
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            else: # opposite direction turning
                this_ya = -self.turn_step/2+c_left_yaw - this_yb-self.yaw_tol # maximum yaw supported
                if this_ya > yaw_after: # maximum turning
                    right_e_y = -self.turn_step/2
                    if yaw_after-this_ya > -self.turn_step/2:
                        # final landing
                        right_e_y = yaw_after-this_ya
                        self.expected_com = np.array(self.last_target)
                        return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                    return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                # final landing
                this_ya = yaw_after
                self.expected_com = np.array(self.last_target)
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
        else: # turning clockwise first
            this_yb = -self.turn_step/2+c_left_yaw-self.yaw_tol # maximum yaw supported
            if this_yb > yaw_before: # still full turning 
                right_e_y = max(yaw_before-this_yb,-self.turn_step/2)
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            this_yb = yaw_before
            # forward moving
            this_d = c_left_x_wy+self.walk_step/2 # maximum step forward supported
            if this_d + self.walk_step/2 < distance: # full speed forwarding
                right_e_x =self.walk_step/2
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            this_d = min(this_d,distance)
            right_e_x = distance-this_d # small half step
            # turn at the end
            if yaw_after <= 0: # same direction turning
                this_ya = -self.turn_step/2+c_left_yaw - this_yb-self.yaw_tol # maximum yaw supported
                if this_ya > yaw_after: # maximum turning
                    right_e_y = -self.turn_step/2
                    if yaw_after-this_ya > -self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        right_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                    return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
            else: # opposite direction turning
                this_ya = self.turn_step/2+c_left_yaw - this_yb+self.yaw_tol # maximum yaw supported
                if this_ya < yaw_after: # maximum turning
                    right_e_y = self.turn_step/2
                    if yaw_after-this_ya < self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        right_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                    return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(right_e_x,right_e_y)
    
    # the expected yaw before forward, forward distance and yaw_after for com
    # the current right x with yaw, current right yaw
    def left_move_one(self,yaw_before,distance,yaw_after,c_right_x_wy,c_right_yaw):
        this_yb,this_d,this_ya = 0,0,0
        left_e_x,left_e_y = 0,0
        if yaw_before >= 0: # turning anti clockwise first
            this_yb = self.turn_step/2+c_right_yaw+ self.yaw_tol # maximum yaw supported
            if this_yb < yaw_before: # still full turning 
                left_e_y = min(yaw_before-this_yb,self.turn_step/2)
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            this_yb = yaw_before
            # forward moving
            this_d = c_right_x_wy+self.walk_step/2 # maximum step forward supported
            if this_d + self.walk_step/2 < distance: # full speed forwarding
                left_e_x =self.walk_step/2
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            this_d = min(this_d,distance)
            left_e_x = distance-this_d # small half step
            # turn at the end
            if yaw_after >= 0: # same direction turning
                this_ya = self.turn_step/2+c_right_yaw - this_yb+self.yaw_tol # maximum yaw supported
                if this_ya < yaw_after: # maximum turning
                    left_e_y = self.turn_step/2
                    if yaw_after-this_ya < self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        left_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                    return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            else: # opposite direction turning
                this_ya = -self.turn_step/2+c_right_yaw - this_yb-self.yaw_tol # maximum yaw supported
                if this_ya > yaw_after: # maximum turning
                    left_e_y = -self.turn_step/2
                    if yaw_after-this_ya > -self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        left_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                    return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
        else: # turning clockwise first
            this_yb = -self.turn_step/2+c_right_yaw-self.yaw_tol # maximum yaw supported
            if this_yb > yaw_before: # still full turning 
                left_e_y = max(yaw_before-this_yb,-self.turn_step/2)
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            this_yb = yaw_before
            # forward moving
            this_d = c_right_x_wy+self.walk_step/2 # maximum step forward supported
            if this_d + self.walk_step/2 < distance: # full speed forwarding
                left_e_x =self.walk_step/2
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            this_d = min(this_d,distance)
            left_e_x = distance-this_d # small half step
            # turn at the end
            if yaw_after <= 0: # same direction turning
                this_ya = -self.turn_step/2+c_right_yaw - this_yb-self.yaw_tol # maximum yaw supported
                if this_ya > yaw_after: # maximum turning
                    left_e_y = -self.turn_step/2
                    if yaw_after-this_ya > -self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        left_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                    return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
            else: # opposite direction turning
                this_ya = self.turn_step/2+c_right_yaw - this_yb+self.yaw_tol # maximum yaw supported
                if this_ya < yaw_after: # maximum turning
                    left_e_y = self.turn_step/2
                    if yaw_after-this_ya < self.turn_step/2:
                        # final landing
                        self.expected_com = np.array(self.last_target)
                        left_e_y = yaw_after-this_ya
                        return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                    return (this_yb,this_d,this_ya),(left_e_x,left_e_y)
                # final landing
                self.expected_com = np.array(self.last_target)
                this_ya = yaw_after
                return (this_yb,this_d,this_ya),(left_e_x,left_e_y)

    # do one step, from stationary, from turning, from walking
    def schedule_step(self,yaw_before,distance,yaw_after):
        # leg switching
        if self.leg_state == "moving_right":
            self.leg_state = "moving_left"
        elif self.leg_state == "moving_left":
            self.leg_state = "moving_right"
        if self.leg_state == "standing":
            # turn right or one step away from turn right
            if yaw_before<0 or (yaw_before<self.yaw_tol and distance<self.walk_step/2+self.loc_tol and yaw_after<0):
                self.leg_state = "moving_right"
            # turn left or one step away from turn left
            elif yaw_before>0 or (yaw_before>-self.yaw_tol and distance<self.walk_step/2+self.loc_tol and yaw_after>0):
                self.leg_state = "moving_left"
            elif self.default_right:
                self.leg_state = "moving_right"
            else:
                self.leg_state = "moving_left"
        # get from previous status
        ref,current_state = self.ik.get_current_target()
        if self.leg_state == "moving_right":
            c_left_x_wy = current_state["left_foot_target"].wxyz_xyz[4] # stationary leg x ahead of com
            c_left_yaw = current_state["left_foot_target"].rotation().compute_yaw_radians()  # stationary leg yaw
            dcom,er = self.right_move_one(yaw_before,distance,yaw_after,c_left_x_wy,c_left_yaw)
            # print(dcom,er)
            # right foot interpolation
            pyaw = current_state["right_foot_target"].rotation().compute_yaw_radians()
            px_wy = current_state["right_foot_target"].wxyz_xyz[4]
            f_t,self._motion["right_foot_target"] = self.move_leg(pyaw,px_wy,er[1],er[0],False)
            # rotate com ref
            self._motion["com_target"] = []
            current_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw),np.array([self.pos_x,self.pos_y,self.walking_com_height]))
            self._motion["com_target"].append([0,current_ref])
            final_x = self.pos_x+math.cos(self.yaw+dcom[0])*dcom[1]
            final_y = self.pos_y+math.sin(self.yaw+dcom[0])*dcom[1]
            end_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw+dcom[0]+dcom[2]),np.array([final_x,final_y,self.walking_com_height]))
            self._motion["com_target"].append([f_t,end_ref])
            # left foot not moving on the ground
            self._motion["left_foot_target"]= []
            self._motion["left_foot_target"].append([0,current_state["left_foot_target"]])
            # ref@tar = ref'@tar'
            # tar' = inv(ref')@ref@tar
            l_end = end_ref.inverse()@current_ref@current_state["left_foot_target"]
            self._motion["left_foot_target"].append([f_t,l_end])
            self.t_horizon = f_t
        else:
            c_right_x_wy = current_state["right_foot_target"].wxyz_xyz[4] # stationary leg x ahead of com
            c_right_yaw = current_state["right_foot_target"].rotation().compute_yaw_radians()  # stationary leg yaw
            dcom,el = self.left_move_one(yaw_before,distance,yaw_after,c_right_x_wy,c_right_yaw)
            # print(dcom,el)
            # left foot interpolation
            pyaw = current_state["left_foot_target"].rotation().compute_yaw_radians()
            px_wy = current_state["left_foot_target"].wxyz_xyz[4]
            f_t,self._motion["left_foot_target"] = self.move_leg(pyaw,px_wy,el[1],el[0],True)
            # rotate com ref
            self._motion["com_target"] = []
            current_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw),np.array([self.pos_x,self.pos_y,self.walking_com_height]))
            self._motion["com_target"].append([0,current_ref])
            final_x = self.pos_x+math.cos(self.yaw+dcom[0])*dcom[1]
            final_y = self.pos_y+math.sin(self.yaw+dcom[0])*dcom[1]
            end_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw+dcom[0]+dcom[2]),np.array([final_x,final_y,self.walking_com_height]))
            self._motion["com_target"].append([f_t,end_ref])
            # right foot not moving on the ground
            self._motion["right_foot_target"]= []
            self._motion["right_foot_target"].append([0,current_state["right_foot_target"]])
            # ref@tar = ref'@tar'
            # tar' = inv(ref')@ref@tar
            r_end = end_ref.inverse()@current_ref@current_state["right_foot_target"]
            self._motion["right_foot_target"].append([f_t,r_end])
            self.t_horizon = f_t


    def schedule_finish(self):
        # get from previous status
        ref,current_state = self.ik.get_current_target()
        if self.leg_state == "moving_left": # previously left moved, now finish right foot
            # right leg
            pyaw = current_state["right_foot_target"].rotation().compute_yaw_radians()
            px_wy = current_state["right_foot_target"].wxyz_xyz[4]
            f_t,self._motion["right_foot_target"] = self.move_leg(pyaw,px_wy,0,0,False)
            # left foot not moving on the ground
            self._motion["left_foot_target"]= []
            self._motion["left_foot_target"].append([0,current_state["left_foot_target"]])
            l_end = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),np.array([0,self.standing_foot_y,-self.walking_com_height]))
            self._motion["left_foot_target"].append([f_t,l_end])
            # ref@tar = ref'@tar'
            # ref' = ref@tar@inv(tar')
            current_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw),np.array([self.pos_x,self.pos_y,self.walking_com_height]))
            end_ref = current_ref@current_state["left_foot_target"]@l_end.inverse()
            self._motion["com_target"] = []
            self._motion["com_target"].append([0,current_ref])
            self._motion["com_target"].append([f_t,end_ref])
            self.t_horizon = f_t
            self.leg_state = "standing"
        else: # previously right moved, now finish left foot
            # left leg
            pyaw = current_state["left_foot_target"].rotation().compute_yaw_radians()
            px_wy = current_state["left_foot_target"].wxyz_xyz[4]
            f_t,self._motion["left_foot_target"] = self.move_leg(pyaw,px_wy,0,0,True)
            # right foot not moving on the ground
            self._motion["right_foot_target"]= []
            self._motion["right_foot_target"].append([0,current_state["right_foot_target"]])
            r_end = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),np.array([0,-self.standing_foot_y,-self.walking_com_height]))
            self._motion["right_foot_target"].append([f_t,r_end])
            # ref@tar = ref'@tar'
            # ref' = ref@tar@inv(tar')
            current_ref = mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,self.yaw),np.array([self.pos_x,self.pos_y,self.walking_com_height]))
            end_ref = current_ref@current_state["right_foot_target"]@r_end.inverse()
            self._motion["com_target"] = []
            self._motion["com_target"].append([0,current_ref])
            self._motion["com_target"].append([f_t,end_ref])
            self.t_horizon = f_t
            self.leg_state = "standing"
    
    def place(self,target):
        # time update
        self.t += self.dt
        if self.t >= self.t_horizon:
            self.t = 0
            if self.arm_moving_flag:
                # expected to finished place
                self.arm_moving_flag = False
                self.t_horizon = 0
                self._motion["right_palm_target"]=[[0,mink.SE3(np.array([0.71048,-0.07177,0.69644,-0.07097,-0.04929,-0.24195,-0.07268]))]]
                return True
            else:
                self.arm_moving_flag = True
                self.t_horizon = 2
                self._motion["right_palm_target"]=[]
                # up
                self._motion["right_palm_target"].append([0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,-0.2,0),np.array([0.3,-0.25,0.4]))])
                # down
                self._motion["right_palm_target"].append([0.5,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),np.array(target))])
                # side
                self._motion["right_palm_target"].append([1,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,-1.2),np.array([0.1,-0.45,0.1]))])
                # drop
                self._motion["right_palm_target"].append([2,mink.SE3(np.array([0.71048,-0.07177,0.69644,-0.07097,-0.04929,-0.24195,-0.07268]))])
        # interpret and move robot
        d_target = self.dance_interpret(self._motion,self.t)
        # update with inverse kinematics
        self.ik.ik(d_target["com_target"],d_target)
        return False

    def pick(self,target):
        # time update
        self.t += self.dt
        if self.t >= self.t_horizon:
            self.t = 0
            if self.arm_moving_flag:
                # expected to finished place
                self.arm_moving_flag = False
                self.t_horizon = 0
                self._motion["right_palm_target"]=[[0,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,-0.2,0),np.array([0.3,-0.25,0.4]))]]
                return True
            else:
                self.arm_moving_flag = True
                self.t_horizon = 2
                self._motion["right_palm_target"]=[]
                # default
                self._motion["right_palm_target"].append([0,mink.SE3(np.array([0.71048,-0.07177,0.69644,-0.07097,-0.04929,-0.24195,-0.07268]))])
                # side up
                self._motion["right_palm_target"].append([1,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,-1.2),np.array([0.1,-0.45,0.2]))])
                # catch
                self._motion["right_palm_target"].append([1.6,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,0,0),np.array(target))])
                # lift
                self._motion["right_palm_target"].append([2,mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(0,-0.2,0),np.array([0.3,-0.25,0.4]))])
        # interpret and move robot
        d_target = self.dance_interpret(self._motion,self.t)
        # update with inverse kinematics
        self.ik.ik(d_target["com_target"],d_target)
        return False


    def dance_interpret(self,instruct,t):
        data = {}
        for k,critical_pose in instruct.items():
            for i in range(len(critical_pose)):
                ct0 = critical_pose[i][0]
                if i < len(critical_pose)-1 and critical_pose[i+1][0] <= t: # wait for t passed
                    continue
                elif i == len(critical_pose)-1: # stay last order
                    data[k] = critical_pose[i][1]
                    break
                else:
                    ct1 = critical_pose[i+1][0]
                    pose = self.se3mix(critical_pose[i][1],t-ct0,critical_pose[i+1][1],ct1-t)
                    data[k] = pose
                    break
        return data

    # se3 pose 1, ratio of pose 1 in mix, se3 pose 2, ratio of pose2 in mix
    def se3mix(self,p1,r1,p2,r2):
        return mink.SE3((np.array(p1.wxyz_xyz)*r2+np.array(p2.wxyz_xyz)*r1)/(r1+r2)) # linear interpolation


if __name__ == "__main__":
    F = 200.0
    rate = RateLimiter(frequency=F, warn=False)
    # ikmodel = unitree_g1_ik("model/unitree_g1/unitree_g1_scene.xml",init_frame="init_frame",additional_constraints=["pelvis_orientation_task","torso_orientation_task"],dt =  rate.dt)
    ikmodel = bxi_elf3_ik("model/bxi_elf3/bxi_elf3_scene.xml",init_frame="init_frame",additional_constraints=["pelvis_orientation_task","torso_orientation_task"],dt =  rate.dt)
    humanoid = humanoid_Function(ikmodel,init_loc=(0,0,0),dt = rate.dt)

    tasks = [
        ("wait",0.5),
        ("walk_to",(-0.5,-0.5,-1)),
        ("wait",0.5),
        ("pick",(0.3,-0.25,0.2)),
        ("wait",0.2),
        ("walk_to",(-0.5,0.5,1)),
        ("wait",0.5),
        ("place",(0.3,-0.25,0.1)),
        ("wait",0.1),
    ]

    # visualization
    with mujoco.viewer.launch_passive(
        model=humanoid.model, data=humanoid.configuration.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(humanoid.model, viewer.cam)
        task_i = 0
        wait_t = 0
        while viewer.is_running():

            mujoco.mj_camlight(humanoid.model, humanoid.configuration.data)

            the_task = tasks[task_i]
            if the_task[0] == "walk_to":
                result = humanoid.walk_to(the_task[1])
            elif the_task[0] == "wait":
                result = wait_t>=the_task[1]
                wait_t += rate.dt
            elif the_task[0] == "pick":
                result = humanoid.pick(the_task[1])
            elif the_task[0] == "place":
                result = humanoid.place(the_task[1])
            
            if result and task_i<len(tasks)-1:
                task_i += 1
                wait_t = 0
            

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(humanoid.model, humanoid.configuration.data)
            mujoco.mj_sensorPos(humanoid.model, humanoid.configuration.data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
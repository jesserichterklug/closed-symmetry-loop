import tensorflow as tf
import nbimporter
from xnp.xnp_numpy import setOfRotations, RToRBar
from utils import *

class MtMCreation(tf.keras.layers.Layer):
    def __init__(self, information_type = "po", strides=1,
                 **kwargs):
        super(MtMCreation, self).__init__(**kwargs)
        self.supports_masking = True
        
        assert(information_type in ["po", "depth"])
        self.information_type = information_type
        self.strides = strides
        
    def call(self,  x):
        if self.information_type == "po":
            obj, w_pixel, camera_matrix_input, coord_K = x
            w_pixel = tf.reshape(w_pixel, (tf.shape(w_pixel)[0],  tf.shape(w_pixel)[1], tf.shape(w_pixel)[2], tf.shape(w_pixel)[3],2,2))   
            return self.makeMtMPersp(obj, w_pixel, camera_matrix_input, coord_K)
    
        if self.information_type == "depth":
            obj, depth, w_depth = x
            return self.makeMtMDepth(tf.ones_like(w_depth), w_depth, depth, obj)
    
        assert(False)
        
    def get_config(self):
        config = {
            'information_type': self.information_type,
            'strides' : self.strides
        }
        base_config = super(MtMCreation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def matrixA (self, obj_image, TCR = None):
        if TCR == None:
            TCR = tf.eye(4, batch_shape=[tf.shape(obj_image)[0]])

        '''Implementation of the matrix A from the above equation'''
        p1 = obj_image[:,:,:,:,0,tf.newaxis]
        p2 = obj_image[:,:,:,:,1,tf.newaxis]
        p3 = obj_image[:,:,:,:,2,tf.newaxis]
        one = tf.ones_like(p1)

        TCR1 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,0]
        TCR2 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,1]
        TCR3 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,2]
        TCR4 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,3]

        return tf.stack(( p1*TCR1,  p2*TCR1,  p3*TCR1, 
                          p1*TCR2,  p2*TCR2,  p3*TCR2, 
                          p1*TCR3,  p2*TCR3,  p3*TCR3, 
                         one*TCR4,
                         one*TCR1, one*TCR2, one*TCR3), axis=-1)

    def matrixAdash (self, objdash_image, TCR = None):
        if TCR == None:
            TCR = tf.eye(4, batch_shape=[tf.shape(objdash_image)[0]])

        one = tf.ones_like(objdash_image[:,:,:,:,0:1])
        zero = tf.zeros_like(one)

        TCR1 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,0]
        TCR2 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,1]
        TCR3 = TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,2]

        return tf.stack(( zero*TCR1,  zero*TCR1,  zero*TCR1, 
                          zero*TCR2,  zero*TCR2,  zero*TCR2, 
                          zero*TCR3,  zero*TCR3,  zero*TCR3, 
                         tf.einsum('bij,bxyoj->bxyoi',TCR[:,:,:3],objdash_image) + TCR[:,tf.newaxis,tf.newaxis,tf.newaxis,:,3],
                         one*TCR1, one*TCR2, one*TCR3), axis=-1)

    def matrixB (self, shape, camera_matrix, coord_K):
#         u, v = tf.meshgrid(tf.range(shape[1], dtype=tf.float32), tf.range(shape[0], dtype=tf.float32))
#         u, v = (u * coord_K[:,0:1,0:1] * self.strides + coord_K[:,1:2,0:1], v * coord_K[:,0:1,1:2] * self.strides + coord_K[:,1:2,1:2])
        u, v = generate_px_coordinates(shape, coord_K, self.strides)
        coords = tf.stack([u, v], axis=-1)[..., tf.newaxis]
        
        # [-f,  0]
        # [ 0, -f]
        left4 = -tf.eye(2,batch_shape=tf.shape(coords)[:-2]) * camera_matrix[:,tf.newaxis,tf.newaxis,:2,:2]
        d1_2 = coords - camera_matrix[:,tf.newaxis,tf.newaxis,:2,2:3] 

        #np.array([[-f,  0, d1, 0],
        #          [ 0, -f, d2, 0]])
        return tf.concat([left4, d1_2, tf.zeros_like(d1_2)], axis=-1)


    def MRowImagePersp (self, obj_image, pixel_weight_image, camera_matrixs,coord_K, Tcr = None):
        '''Implements the whole above equation taking one entry of the dataset including object point,
        image point and weighting matrix. '''

        W = pixel_weight_image
        A = self.matrixA (obj_image, Tcr)
        B = self.matrixB (obj_image.shape[1:3],camera_matrixs, coord_K)
    #     print("WAB", W, A, B)

        BA = tf.einsum('bxyij,bxyojk->bxyoik', B, A) #tf.matmul(B, A)
        MRow = tf.matmul(W, BA)
    #     print("BA, MRow", BA, MRow)

        return MRow


    def makeMtMPersp (self, obj_image, pixel_weight_image, camera_matrixs, coord_K, Tcr = None):
        '''Computes the Information Matrix MtM from the input data.
           This matrix has size 13^2 regardless of the number of points n.
           Here \bar T MtM bar T is the optimization problem to be solved.'''
        M = self.MRowImagePersp (obj_image, pixel_weight_image, camera_matrixs, coord_K, Tcr=Tcr)
        num_of_equations = 2 * obj_image.shape[1] * obj_image.shape[2]
        M = tf.reshape(tf.transpose(M, perm=[0,3,1,2,4,5]), (tf.shape(obj_image)[0], obj_image.shape[3], num_of_equations,  13))
        M = tf.dtypes.cast(M, tf.float64)
        return tf.matmul(M,M, transpose_a=True)
    
    
    def matrixC (self, w_depth_meassurement, w_depth_object , z):
        '''Matrix B (see above)'''
        w1 = w_depth_meassurement**2
#         print('makeMtMDepth','matrixC','w1',w1.shape)
        w2 = w_depth_object**2
#         print('makeMtMDepth','matrixC','w2',w2.shape)
        w = (w1*w2 / (w1 + w2 + 0.0001))**0.5
        z = tf.expand_dims(tf.expand_dims(z,axis=-1),axis=-1)
#         print('makeMtMDepth','matrixC','w',w.shape)
#         print('makeMtMDepth','matrixC','z',z.shape)
        zero = tf.zeros_like(w)
#         print('makeMtMDepth','matrixC','zero',zero.shape)
        C = tf.stack([zero, zero, -w*z, w*z**2],axis=-1)#[:,:,:,:,tf.newaxis]
        return C

    def MRowDepth (self, w_depth_meassurement, w_depth_object, z_coord_image, obj_image):
        '''Implements the whole above equation taking one entry of the depth dataset'''
        A = self.matrixA(obj_image)
        C = self.matrixC(w_depth_meassurement, w_depth_object , z_coord_image)

#         print('MRowDepth','C', C.shape)
#         print('MRowDepth','A', A.shape)

        return tf.matmul(C, A)


    def makeMtMDepth (self, pixel_weight_depth_meassurement, pixel_weight_d_obj, z_coord_image, obj_image):
        '''Computes the Information Matrix MtM from the input data.
           This matrix has size 13^2 regardless of the number of points n.
           Here \bar T MtM bar T is the optimization problem to be solved.'''

        M = self.MRowDepth (pixel_weight_depth_meassurement, pixel_weight_d_obj, z_coord_image, obj_image)
#         print('makeMtMDepth','M_',M.shape)

        num_of_equations = 1 * obj_image.shape[1] * obj_image.shape[2]
        M = tf.reshape(tf.transpose(M, perm=[0,3,1,2,4,5]), (tf.shape(obj_image)[0], obj_image.shape[3], num_of_equations,  13))
        M = tf.dtypes.cast(M, tf.float64)

#         print('makeMtMDepth','M',M.shape)
        MtMDepth = tf.matmul(M,M, transpose_a=True)
        return MtMDepth
    
class DirectCalcT_viaDash(tf.keras.layers.Layer):
    def __init__(self, 
                 **kwargs):
        super(DirectCalcT_viaDash, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, MtMs, po, vo, seg, w, cam_K, coord_K):
        
        R = self.direct_calc_R(po, vo, seg, w)   
        new_dash = tf.einsum('bij, byxj -> byxi', R, po[...,0,:] * seg)
        t = self.direct_calc_t(vo, w, cam_K, coord_K, seg)
        
        Rt = tf.concat([R,t], axis=-1)
        _0001 = tf.concat([tf.zeros_like(R[...,:1,:]),tf.ones_like(t[...,:1,:])], axis=-1)
        return tf.concat([Rt,_0001], axis=-2)[...,tf.newaxis,:,:]
        
    def direct_calc_R(self, po, vo, seg, w):
        
        w = tf.reshape(w, [tf.shape(w)[0], w.shape[1], w.shape[2], 2,2]) * seg[..., tf.newaxis]
        w = tf.reduce_sum(tf.matmul(w, tf.ones_like(w[...,:1])), axis=[-2], keepdims=True)
        poseg = po[...,0,:] * seg
        voseg = vo * seg
        
        def direct_calc_axis(axis, po_segmented, vo_segmented, w):
            A = vo_segmented[...,tf.newaxis,:] * w
            AtA = tf.matmul(A,A, transpose_a=True)
            b = po_segmented[...,axis:axis+1,tf.newaxis] * w
            Atb = tf.matmul(A,b, transpose_a=True)

            _AtA = tf.reduce_sum(AtA,axis=[1,2])
            _Atb = tf.reduce_sum(Atb,axis=[1,2])

            return tf.matmul(tf.linalg.pinv(_AtA), _Atb)[...,0]   
        
        #x = direct_calc_axis(0, poseg, voseg, w)
        y = direct_calc_axis(1, poseg, voseg, w)
        z = direct_calc_axis(2, poseg, voseg, w)

        x = cross(y,z)
        y = cross(z,x)
        z = cross(x,y)

        return tf.stack([x,y,z], axis=-1)
    
    def direct_calc_t(self, vo, w, cam_K, coord_K, seg):    
    
        def make_equation_matrix_pd(vo, b, k, w):
            A = tf.matmul(w, k)
            AtA = tf.matmul(A,A, transpose_a=True)
            b = tf.matmul(w, b - tf.matmul(k, vo[..., tf.newaxis]))
            Atb = tf.matmul(A,b, transpose_a=True)
            return AtA, Atb
            
        K = self.matrixB(vo.shape[1:3], cam_K, coord_K)
        w = tf.reshape(w, [tf.shape(w)[0], w.shape[1], w.shape[2], 2,2]) * seg[..., tf.newaxis]
        AtA, Atb = make_equation_matrix_pd(vo, 0, K, w)
    
        _AtA = tf.reduce_sum(AtA,axis=[1,2])
        _Atb = tf.reduce_sum(Atb,axis=[1,2])
        
        return tf.matmul(tf.linalg.pinv(_AtA), _Atb)    
    
    def matrixB (self, shape, camera_matrix, coord_K):
        u, v = tf.meshgrid(tf.range(shape[1], dtype=tf.float32), tf.range(shape[0], dtype=tf.float32))
        u, v = (u * coord_K[:,0:1,0:1] + coord_K[:,1:2,0:1], v * coord_K[:,0:1,1:2] + coord_K[:,1:2,1:2])
        coords = tf.stack([u, v], axis=-1)[..., tf.newaxis]
        
        # [-f,  0]
        # [ 0, -f]
        left4 = -tf.eye(2,batch_shape=tf.shape(coords)[:-2]) * camera_matrix[:,tf.newaxis,tf.newaxis,:2,:2]
        d1_2 = coords - camera_matrix[:,tf.newaxis,tf.newaxis,:2,2:3] 

        #np.array([[-f,  0, d1],
        #          [ 0, -f, d2]])
        return tf.concat([left4, d1_2], axis=-1)
        
        
class MtMEvaluation(tf.keras.layers.Layer):
    def __init__(self, 
                 **kwargs):
        super(MtMEvaluation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.rotationSetBarTensor = tf.constant([RToRBar(R) for R in setOfRotations()])[:,:,tf.newaxis]
    
    def call(self, MTMs, Ts=None):
        if Ts is None:
            Ts = self.initialGuessRbased(MTMs)
        seg_counts= None

        Ts, Omega = self.pnpRefineMtM(MTMs,Ts,seg_counts)
        Ts = tf.debugging.assert_all_finite(Ts, 'T is not finit')

        Ts, Omega = self.pnpRefineMtM(MTMs,Ts,seg_counts)
        Ts = tf.debugging.assert_all_finite(Ts, 'T is not finit')

        Ts, Omega = self.pnpRefineMtM(MTMs,Ts,seg_counts)
        Ts = tf.debugging.assert_all_finite(Ts, 'T is not finit')

        Ts, Omega = self.pnpRefineMtM(MTMs,Ts,seg_counts)
        Ts = tf.debugging.assert_all_finite(Ts, 'T is not finit')

        Ts, Omega = self.pnpRefineMtM(MTMs,Ts,seg_counts)
        Ts = tf.debugging.assert_all_finite(Ts, 'T is not finit')
        return Ts, Omega
        
        
    def POfT (self, Tbreve):
        '''Computes the matrix P(TBreve) that maps a vector (rx,ry,rz,tx,ty,tz,1) to the vector (bar)
           representation of RBreve times differential transform of (rx,ry,rz,tx,ty,tz)'''
        zero = tf.zeros_like(Tbreve[:,:,0,0])
        one = tf.ones_like(Tbreve[:,:,0,0])
        return tf.stack([
            tf.stack([    zero    , -Tbreve[:,:,0,2], Tbreve[:,:,0,1], zero,zero,zero, Tbreve[:,:,0,0]], axis=-1),
            tf.stack([ Tbreve[:,:,0,2],    zero    , -Tbreve[:,:,0,0], zero,zero,zero, Tbreve[:,:,0,1]], axis=-1),
            tf.stack([-Tbreve[:,:,0,1],  Tbreve[:,:,0,0],    zero    , zero,zero,zero, Tbreve[:,:,0,2]], axis=-1),
            #
            tf.stack([    zero    , -Tbreve[:,:,1,2], Tbreve[:,:,1,1], zero,zero,zero, Tbreve[:,:,1,0]], axis=-1),
            tf.stack([ Tbreve[:,:,1,2],    zero    , -Tbreve[:,:,1,0], zero,zero,zero, Tbreve[:,:,1,1]], axis=-1),
            tf.stack([-Tbreve[:,:,1,1],  Tbreve[:,:,1,0],    zero    , zero,zero,zero, Tbreve[:,:,1,2]], axis=-1),
            #
            tf.stack([    zero    , -Tbreve[:,:,2,2], Tbreve[:,:,2,1], zero,zero,zero, Tbreve[:,:,2,0]], axis=-1),
            tf.stack([ Tbreve[:,:,2,2],    zero    , -Tbreve[:,:,2,0], zero,zero,zero, Tbreve[:,:,2,1]], axis=-1),
            tf.stack([-Tbreve[:,:,2,1],  Tbreve[:,:,2,0],    zero    , zero,zero,zero, Tbreve[:,:,2,2]], axis=-1),
            #
            tf.stack([    zero    ,    zero    ,    zero    ,    zero    ,    zero    ,    zero    , one], axis=-1),
            #
            tf.stack([ zero,zero,zero, Tbreve[:,:,0,0], Tbreve[:,:,0,1], Tbreve[:,:,0,2], Tbreve[:,:,0,3]], axis=-1),
            tf.stack([ zero,zero,zero, Tbreve[:,:,1,0], Tbreve[:,:,1,1], Tbreve[:,:,1,2], Tbreve[:,:,1,3]], axis=-1),
            tf.stack([ zero,zero,zero, Tbreve[:,:,2,0], Tbreve[:,:,2,1], Tbreve[:,:,2,2], Tbreve[:,:,2,3]], axis=-1)
            ], axis=-2)



    def computeQT (self, MtM, Tbreve):
        '''Computes the 7*7 matrix Q describing the 2nd order optimization problem obtained from
           linearizing the transform relative to Tbreve in the optimization problem described by MtM
           as a 13*13 matrix. See explanation above.

           The part [0:6,0:6] of the result is also the information matrix (inverse covariance)
           of the transform uncertainty when using the [+]:r times delta --> r*exp(hat(delta))
           operator assuming the least-squares system Ptilde is properly scaled (1 meaning
           unit noise)'''
        myPOfT = self.POfT (Tbreve)
        return tf.matmul(tf.matmul(myPOfT, MtM, transpose_a=True), myPOfT)

    def solveQT (self, Q):
        '''Computes min_{v_last=1} v^T Q v and returns v, the minimal value obtained and inverse covariance'''
        regR = tf.ones_like(Q[:,:,0,0]) * 10 # regularizer corresponding to a std. dev. of sqrt(3/10) radian \approx 31deg
                  # this is a small damping and makes sense even from a linearization perspective
        regT = 1E-8*tf.maximum(Q[:,:,3,3]+Q[:,:,4,4]+Q[:,:,5,5], tf.ones_like(Q[:,:,0,0]))
                  # the regularizer in translation needs to be proportional to the (squared) scale
                  # of translation. This is obtained by the trace of the translation part of Q
                  # If that's 0, we still need to make the matrix SPD to avoid failure, hence the
                  # max (...., 1)

        Omega = Q[:,:,0:6,0:6] + tf.linalg.diag(tf.stack([regR, regR, regR, regT, regT, regT],axis=-1))

        #print (np.linalg.det(Q[0:6,0:6]), np.linalg.det(Omega))
        v = -tf.matmul(self.inv(Omega), Q[:,:,0:6,6:7])
        v = tf.concat([v, tf.ones_like(v[:,:,:1,:1])], axis=-2)
        newVal = tf.matmul(tf.matmul(v,Q,transpose_a=True),v)
        return (v, newVal, Omega)

    def optimizeTIteration (self, MtM, Tbreve):
        '''Performs one linearized optimization step on min_{T in SE(3)} \bar T^T MtM \bar T. Tbreve
           is the initial guess / result of the last iteration (as a vector). The function returns the new guess
           which is projected on SE(3), i.e. an element of SE(3), the old function value,
           the linearized function value predicted, the final true function value
           and the inverse covariance.
           Refer to chi2Error or logLikelihood for the meaning of the covariance and note that 
           the covariance is only valid when the iteration has reasonably converged, i.e. the 
           result has not changed too much'''
        Q = self.computeQT (MtM, Tbreve)
        v, linearizedMin, Omega = self.solveQT (Q)
        TNew = self.boxplus(Tbreve, v)
        TNewBar = self.TToTBar (TNew)
        newVal = tf.matmul(TNewBar, tf.matmul(MtM, TNewBar), transpose_a=True)
        return TNew, newVal, Omega

    def pnpRefineMtM (self, MtM, Tbreve, lendata = None):
        #explicit cast because keras-layer-shapeinferenc will break otherwise
        MtM = tf.dtypes.cast(MtM, tf.float64)
        Tbreve = tf.dtypes.cast(Tbreve, tf.float64)

        T, val, Omega = self.optimizeTIteration (MtM, Tbreve)
        Omega = tf.debugging.assert_all_finite(Omega, 'Omega is not finite')

        if not lendata is None:
            lendata = tf.dtypes.cast(lendata, tf.float64)

            ms = val/ (lendata[:,:,tf.newaxis]+0.00000001) +0.00000001 #rms²
            Omega /= ms 
            Omega = tf.debugging.assert_all_finite(Omega, 'Omega/ms is not finite')
        else:
            # As explained in the derivation the measurement equations are multiplied
            # by the respective point's z-camera-coordinate, hence we devide here
            # by the object's orgin's z-camera-coordinate.
            Omega = tf.debugging.assert_all_finite(Omega/(T[:,:,2:3,3:4]**2+0.00000001), 'Omega2 is not finite')
        return (T, Omega) 


    def logm_approx(self, M):
        return M


    def inv(self, M):
        M = tf.debugging.assert_all_finite(M, 'inv entry is not finit')
        M += tf.eye(tf.shape(M)[-1], dtype=M.dtype) * 0.00001 \
                * (tf.constant(1,M.dtype) + tf.reduce_max(M,axis=[-2,-1],keepdims=True))
        M = tf.debugging.assert_all_finite(M, 'inv middle is not finit')
        M_inv = tf.linalg.inv(M,name='the_inv')
        M_inv = tf.debugging.assert_all_finite(M_inv, 'inv closure is not finit')
        return M_inv
    
    
    def TToTBar (self, T):
        '''Converts a SE(3) matrix T into a stacked vector \bar T according to the above stacking convention'''
        one = tf.ones_like(T[:,:,0,0])
        return tf.stack([ T[:,:,0,0], T[:,:,0,1], T[:,:,0,2], T[:,:,1,0], T[:,:,1,1], T[:,:,1,2], 
                         T[:,:,2,0], T[:,:,2,1], T[:,:,2,2], one , T[:,:,0,3], T[:,:,1,3], T[:,:,2,3] 
                    ], axis=-1)[:,:,:,tf.newaxis]


    def diffT (self, v):
        '''Returns the differential transform parametrized by v in R^6.'''
        zero = tf.zeros_like(v[:,:,0,0])
        return tf.stack([tf.stack([   zero,-v[:,:,2,0], v[:,:,1,0], v[:,:,3,0]],axis=-1),
                         tf.stack([ v[:,:,2,0],   zero,-v[:,:,0,0], v[:,:,4,0]],axis=-1),
                         tf.stack([-v[:,:,1,0], v[:,:,0,0],   zero, v[:,:,5,0]],axis=-1),
                         tf.stack([      zero,      zero,      zero,     zero],axis=-1)
                        ],axis=-2)

    def transf (self, v):
        '''Returns the non-differential transform parametrized by v in R^6.'''
        return tf.linalg.expm(self.diffT(v))

    def boxplus (self, T, v):
        '''[+]-operator used for least-square estimating a transform'''
        return tf.matmul(T, self.transf(v))
    # ********************* Global Optimization of the Rotation *******************

    # **************** Schur-Complement ****************************
    def schurComplement (self, A, n):
        ''' Let x be decomposed as x^T=(y,z)^T, with dim y = n. Then schurComplement
        decomposes the quadratic function x^TAx as y^T Ptilde y + (Hy-z)^T S (Hy-z).
        This means, y^T Stilde y is the optimization problem for y only, with z
        removed optimally and $z=Hy$ is the corresponding optimal z for a given y. S
        specifies how much a deviation from the optimum increases the cost function.
        The function returns (Ptilde, H, S).'''
        #np.testing.assert_almost_equal (A, A.T)
        # A.shape equals [batch, objects (5), ...]
        #m = tf.shape(A)[2 + 0]
        P = A[:,:, :n, :n]
        R = A[:,:, n:, :n]
        S = A[:,:, n:, n:]
        SI = self.inv(S)
        H = -tf.matmul(SI, R)
        Ptilde = P + tf.matmul(R, H, transpose_a = True)
        return (Ptilde, H, S)

    def evaluateR (self, Ptilde, h):

        rbar = tf.cast(self.rotationSetBarTensor, Ptilde.dtype)
        '''Evaluates the const function defined by Ptilde on rbar and checks that h.dot(rbar)>=0'''
    #     print(evaluateR)
    #     print(rbar)
    #     print(h)
        zcCoord = tf.einsum('bcij,sjk->bcsik', h, rbar) # ==> tf.matmul(h[:,:,tf.newaxis], rbar[tf.newaxis,tf.newaxis])
        isBehindCam = tf.cast(zcCoord < tf.constant(0., dtype=h.dtype), h.dtype) * tf.constant(1e+24, h.dtype)

        evaluation_ = tf.einsum('bcij,sjk->bcsik', Ptilde, rbar) # ==> tf.matmul(Ptilde[:,:,tf.newaxis], rbar[tf.newaxis,tf.newaxis])
        evaluation = tf.einsum('sji,bcsjk->bcsik', rbar, evaluation_) # ==> tf.matmul(rbar[tf.newaxis,tf.newaxis], evaluation_ ,transpose_a = True)

        return tf.squeeze(evaluation + isBehindCam, axis = [-2,-1])

    def RBartToT (self, RBar, t):
        zero = tf.zeros_like(RBar[:,:,0,0])
        one  = tf.ones_like(RBar[:,:,0,0])
        '''Converts a stacked vector \bar R and a vector \bar t into a 4*4 matrix according to the above stacking convention'''
        return tf.stack([tf.stack([RBar[:,:,0,0], RBar[:,:,1,0], RBar[:,:,2,0], t[:,:,0,0]],axis=-1),
                         tf.stack([RBar[:,:,3,0], RBar[:,:,4,0], RBar[:,:,5,0], t[:,:,1,0]],axis=-1),
                         tf.stack([RBar[:,:,6,0], RBar[:,:,7,0], RBar[:,:,8,0], t[:,:,2,0]],axis=-1),
                         tf.stack([         zero,          zero,          zero,        one],axis=-1)
                         ],axis=-2)
    

    def initialGuessRbased (self, MtMs):
        '''Provides an initial guess for the rotation by returning the rotation in rotationsSet (roughly 23deg space)
           with the smallest error. It only considers rotations rbar where h.dot(rbar)>=0. This can be used to filter
           out rotations where the resulting translation would be behind the camera.'''
        Ptilde, H, _ = self.schurComplement (MtMs, 10)
        h = H[:,:, 2:3]

        setEvaluationRes = self.evaluateR(Ptilde, h)# [evaluateR(Ptilde, rbar, h) for rbar in rotationSetBar]
        idx = tf.argmin(setEvaluationRes, axis=-1)#:,:,-1]
        #dx = tf.reshape(idx[:,-1], tf.shape(setEvaluationRes)[:2], name='reshape_idx')

        RBar = tf.gather_nd(tf.cast(self.rotationSetBarTensor, Ptilde.dtype), idx[:,:,tf.newaxis])
        t = tf.matmul(H, RBar)

        return self.RBartToT (RBar, t)
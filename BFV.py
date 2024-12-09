# BFV

from poly import *

class BFV:
    # Definitions
    # Z_q[x]/f(x) = x^n + 1 where n=power-of-two

    # Operations
    # -- SecretKeyGen
    # -- PublicKeyGen
    # -- Encryption
    # -- Decryption
    # -- EvaluationKeyGenV1
    # -- EvaluationKeyGenV2 (need to be fixed)
    # -- HomAdd
    # -- HomMult
    # -- RelinV1
    # -- RelinV2 (need to be fixed)

    # Parameters
    # (From outside)
    # -- n (ring size)
    # -- q (ciphertext modulus)
    # -- t (plaintext modulus)
    # -- mu (distribution mean)
    # -- sigma (distribution std. dev.)
    # -- qnp (NTT parameters: [w,w_inv,psi,psi_inv])
    # (Generated with parameters)
    # -- sk
    # -- pk
    # -- rlk1, rlk2

    def __init__(self, n, q, t, mu, sigma, qnp):
        self.n = n
        self.q = q
        self.t = t
        self.T = 0
        self.l = 0
        self.p = 0
        self.mu = mu
        self.sigma = sigma
        self.qnp= qnp # array NTT parameters: [w,w_inv,psi,psi_inv]
        #
        self.sk = []
        self.pk = []
        self.rlk1 = []
        self.rlk2 = []
    #
    def __str__(self):
        str = "\n--- Parameters:\n"
        str = str + "n    : {}\n".format(self.n)
        str = str + "q    : {}\n".format(self.q)
        str = str + "t    : {}\n".format(self.t)
        str = str + "T    : {}\n".format(self.T)
        str = str + "l    : {}\n".format(self.l)
        str = str + "p    : {}\n".format(self.p)
        str = str + "mu   : {}\n".format(self.mu)
        str = str + "sigma: {}\n".format(self.sigma)
        return str
    #
    def SecretKeyGen(self):
        """
        sk <- R_2
        """
        s = Poly(self.n,self.q,self.qnp)
        s.randomize(2)
        self.sk = s
    #
    def PublicKeyGen(self):
        """
        a <- R_q
        e <- X
        pk[0] <- (-(a*sk)+e) mod q
        pk[1] <- a
        """
        a, e = Poly(self.n,self.q,self.qnp), Poly(self.n,self.q,self.qnp)
        a.randomize(self.q)
        e.randomize(0, domain=False, type=1, mu=self.mu, sigma=self.sigma)
        pk0 = -(a*self.sk + e)
        pk1 = a
        self.pk = [pk0,pk1]
    #
    def EvalKeyGenV1(self, T):
        self.T = T
        self.l = int(math.floor(math.log(self.q,self.T)))

        rlk1 = []

        sk2 = (self.sk * self.sk)

        for i in range(self.l+1):
            ai   , ei    = Poly(self.n,self.q,self.qnp), Poly(self.n,self.q,self.qnp)
            ai.randomize(self.q)
            ei.randomize(0, domain=False, type=1, mu=self.mu, sigma=self.sigma)

            Ts2   = Poly(self.n,self.q,self.qnp)
            Ts2.F = [((self.T**i)*j) % self.q for j in sk2.F]

            rlki0 = Ts2 - (ai*self.sk + ei)
            rlki1 = ai

            rlk1.append([rlki0,rlki1])

        self.rlk1 = rlk1
    #
    def Encryption(self, m):
        """
        delta = floor(q/t)

        u  <- random polynomial from R_2
        e1 <- random polynomial from R_B
        e2 <- random polynomial from R_B

        c0 <- pk0*u + e1 + m*delta
        c1 <- pk1*u + e2
        """
        delta = int(math.floor(self.q/self.t))

        u, e1, e2 = Poly(self.n,self.q,self.qnp), Poly(self.n,self.q,self.qnp), Poly(self.n,self.q,self.qnp)

        u.randomize(2)
        e1.randomize(0, domain=False, type=1, mu=self.mu, sigma=self.sigma)
        e2.randomize(0, domain=False, type=1, mu=self.mu, sigma=self.sigma)

        md = Poly(self.n,self.q,self.qnp)
        md.F = [(delta*x) % self.q for x in m.F]

        c0 = self.pk[0]*u + e1
        c0 = c0 + md
        c1 = self.pk[1]*u + e2

        return [c0,c1]
    #
    def Decryption(self, ct):
        """
        ct <- c1*s + c0
        ct <- floot(ct*(t/q))
        m <- [ct]_t
        """
        m = ct[1]*self.sk + ct[0]
        m.F = [((self.t*x)/self.q) for x in m.F]
        m = round(m)
        m = m % self.t
        mr = Poly(self.n,self.t,self.qnp)
        mr.F = m.F
        mr.inNTT = m.inNTT
        return mr
    #
    def RelinearizationV1(self,ct):
        c0 = ct[0]
        c1 = ct[1]
        c2 = ct[2]

        # divide c2 into base T
        c2i = []

        c2q = Poly(self.n,self.q,self.qnp)
        c2q.F = [x for x in c2.F]

        for i in range(self.l+1):
            c2r = Poly(self.n,self.q,self.qnp)

            for j in range(self.n):
                qt = int(c2q.F[j]/self.T)
                rt = c2q.F[j] - qt*self.T

                c2q.F[j] = qt
                c2r.F[j] = rt

            c2i.append(c2r)

        c0r = Poly(self.n,self.q,self.qnp)
        c1r = Poly(self.n,self.q,self.qnp)
        c0r.F = [x for x in c0.F]
        c1r.F = [x for x in c1.F]

        for i in range(self.l+1):
            c0r = c0r + (self.rlk1[i][0] * c2i[i])
            c1r = c1r + (self.rlk1[i][1] * c2i[i])

        return [c0r,c1r]
    #
    def IntEncode(self,m): # integer encode
        mr = Poly(self.n,self.t)
        if m >0:
            mt = m
            for i in range(self.n):
                mr.F[i] = (mt % 2)
                mt      = (mt // 2)
        elif m<0:
            mt = -m
            for i in range(self.n):
                mr.F[i] = (self.t-(mt % 2)) % self.t
                mt      = (mt // 2)
        else:
            mr = mr
        return mr
    #
    def IntDecode(self,m): # integer decode
        mr = 0
        thr_ = 2 if(self.t == 2) else ((self.t+1)>>1)
        for i,c in enumerate(m.F):
            if c >= thr_:
                c_ = -(self.t-c)
            else:
                c_ = c
            mr = (mr + (c_ * pow(2,i)))
        return mr
    #
    def HomomorphicAddition(self, ct0, ct1):
        ct0_b = ct0[0] + ct1[0]
        ct1_b = ct0[1] + ct1[1]
        return [ct0_b,ct1_b]
    #
    def HomomorphicMultiplication(self, ct0, ct1):
        ct00 = ct0[0]
        ct01 = ct0[1]
        ct10 = ct1[0]
        ct11 = ct1[1]

        r0 = RefPolMulv2(ct00.F,ct10.F)
        r1 = RefPolMulv2(ct00.F,ct11.F)
        r2 = RefPolMulv2(ct01.F,ct10.F)
        r3 = RefPolMulv2(ct01.F,ct11.F)

        c0 = [x for x in r0]
        c1 = [x+y for x,y in zip(r1,r2)]
        c2 = [x for x in r3]

        c0 = [((self.t * x) / self.q) for x in c0]
        c1 = [((self.t * x) / self.q) for x in c1]
        c2 = [((self.t * x) / self.q) for x in c2]

        c0 = [(round(x) % self.q) for x in c0]
        c1 = [(round(x) % self.q) for x in c1]
        c2 = [(round(x) % self.q) for x in c2]

        # Move to regular modulus
        r0 = Poly(self.n,self.q,self.qnp)
        r1 = Poly(self.n,self.q,self.qnp)
        r2 = Poly(self.n,self.q,self.qnp)

        r0.F = c0
        r1.F = c1
        r2.F = c2

        return [r0,r1,r2]
    
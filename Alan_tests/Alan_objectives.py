import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe
from Helpers import *

# Penalizes the configuration's maximum elongation
def MaxElongationPen(vmec,t=6.0,ntheta=50,nphi=50):
    """
    Penalizes the configuration's maximum elongation (e_max) if it exceeds some threshold (t).
    Specifically, if e_max > t, then output (e_max - t). Else, output zero.
    vmec        -   VMEC object
    t           -   Mximum elongation above which the output is nonzero
    ntheta      -   Number of points per poloidal cross-section
    nphi        -   Number of poloidal cross-sections
    """
    nfp = vmec.wout.nfp
    # Load variables from VMEC
    if 1 == 1:
        xm = vmec.wout.xm
        xn = vmec.wout.xn
        rmnc = vmec.wout.rmnc.T
        zmns = vmec.wout.zmns.T
        lasym = vmec.wout.lasym
        raxis_cc = vmec.wout.raxis_cc
        zaxis_cs = vmec.wout.zaxis_cs
        if lasym == True:
            raxis_cs = vmec.wout.raxis_cs
            zaxis_cc = vmec.wout.zaxis_cc
            rmns = vmec.wout.rmns
            zmnc = vmec.wout.zmnc
        else:
            raxis_cs = 0*raxis_cc
            zaxis_cc = 0*zaxis_cs
            rmns = rmnc*0
            zmnc = zmns*0

        # Set up variables
        theta1D = np.linspace(0,2*np.pi,num=ntheta)
        phi1D = np.linspace(0,2*np.pi/nfp,num=nphi)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # A function that will return the cartesian coordinates of the boundary for a given pair of VMEC angles
    def FindBoundary(theta,phi):
        phi = phi[0]
        rb = np.sum(rmnc[-1,:] * np.cos(xm*theta + xn*phi))
        zb = np.sum(zmns[-1,:] * np.sin(xm*theta + xn*phi))
        xb = rb * np.cos(phi)
        yb = rb * np.sin(phi)

        return np.array([xb,yb,zb])

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    # Set up axis
    if 1 == 1:
        Rax = np.zeros(nphi)
        Zax = np.zeros(nphi)
        Raxp = np.zeros(nphi)
        Zaxp = np.zeros(nphi)
        Raxpp = np.zeros(nphi)
        Zaxpp = np.zeros(nphi)
        Raxppp = np.zeros(nphi)
        Zaxppp = np.zeros(nphi)
        for jn in range(len(raxis_cc)):
            n = jn * nfp
            sinangle = np.sin(n * phi1D)
            cosangle = np.cos(n * phi1D)

            Rax += raxis_cc[jn] * cosangle
            Zax += zaxis_cs[jn] * sinangle
            Raxp += raxis_cc[jn] * (-n * sinangle)
            Zaxp += zaxis_cs[jn] * (n * cosangle)
            Raxpp += raxis_cc[jn] * (-n * n * cosangle)
            Zaxpp += zaxis_cs[jn] * (-n * n * sinangle)
            Raxppp += raxis_cc[jn] * (n * n * n * sinangle)
            Zaxppp += zaxis_cs[jn] * (-n * n * n * cosangle)

            if lasym == True:
                Rax += raxis_cs[jn] * sinangle
                Zax += zaxis_cc[jn] * cosangle + zaxis_cs[jn] * sinangle
                Raxp += raxis_cs[jn] * (n * cosangle)
                Zaxp += zaxis_cc[jn] * (-n * sinangle)
                Raxpp += raxis_cs[jn] * (-n * n * sinangle)
                Zaxpp += zaxis_cc[jn] * (-n * n * cosangle)
                Raxppp += raxis_cs[jn] * (-n * n * n * cosangle)
                Zaxppp += zaxis_cc[jn] * (n * n * n * sinangle) 

        Xax = Rax * np.cos(phi1D)
        Yax = Rax * np.sin(phi1D)

        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################

        d_l_d_phi = np.sqrt(Rax * Rax + Raxp * Raxp + Zaxp * Zaxp)
        d2_l_d_phi2 = (Rax * Raxp + Raxp * Raxpp + Zaxp * Zaxpp) / d_l_d_phi

        d_r_d_phi_cylindrical = np.array([Raxp, Rax, Zaxp]).transpose()
        d2_r_d_phi2_cylindrical = np.array([Raxpp - Rax, 2 * Raxp, Zaxpp]).transpose()

        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_cylindrical = np.zeros((nphi, 3))
        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_R   = tangent_cylindrical[:,0]
        tangent_phi = tangent_cylindrical[:,1]

        tangent_Z   = tangent_cylindrical[:,2]
        tangent_X   = tangent_R * np.cos(phi1D) - tangent_phi * np.sin(phi1D)
        tangent_Y   = tangent_R * np.sin(phi1D) + tangent_phi * np.cos(phi1D)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # Arrays that will store cross-section locations, for various poloidal angles at a fixed toroidal angle
    Xp = np.zeros(ntheta)
    Yp = np.zeros(ntheta)
    Zp = np.zeros(ntheta)
    # An array that will store the elongations at various toroidal angles
    elongs = np.zeros(nphi)

    # Loop through toroidal angles, finding the elongation of each one, and storing it in elongs
    for iphi in range(nphi):
        phi = phi1D[iphi]

        # x,y,z components of the axis tangent
        tx = tangent_X[iphi]
        ty = tangent_Y[iphi]
        tz = tangent_Z[iphi]
        t_ = np.array([tx,ty,tz])
        # x,y,z location of the axis
        xax = Xax[iphi]
        yax = Yax[iphi]
        zax = Zax[iphi]
        pax = np.array([xax, yax, zax])
        
        # Loop through poloidal angles, keeping toroidal angle fixed
        for ipt in range(ntheta):
            theta = theta1D[ipt]
            # This function returns zero when the point on the boundary is perpendicular to the axis' tangent vector
            fdot = lambda p : np.dot( t_ , (FindBoundary(theta, p) - pax) )
            # Find the cross-section's  point'
            phi_x = fsolve(fdot, phi)
            sbound = FindBoundary(theta, phi_x)
            # Subtract any noise
            sbound -= np.dot(sbound,t_)
            
            # Store cross-section locations
            Xp[ipt] = sbound[0]
            Yp[ipt] = sbound[1]
            Zp[ipt] = sbound[2]
        # Find the perimeter and area the boundary cross-section
        perim = np.sum(np.sqrt((Xp-np.roll(Xp,1))**2 + (Yp-np.roll(Yp,1))**2 + (Zp-np.roll(Zp,1))**2))
        A = FindArea(Xp,Yp,Zp)

        # Area of ellipse = A = pi*a*b
        #   a = semi-major, b = semi-minor
        # b = A / (pi*a)
        # Eccentricity = e = 1 - b**2/a**2
        #                  = 1 - A**2 / (pi**2 * a**4)
        #                  = 1 - (A / (pi * a**2))**2
        # Circumference = C = 4 * a * ellipe(e) --> Use this to solve for semi-major radius a
        #
        # b = A / (pi * a)
        # Elongation = E = semi-major / semi-minor 
        #                = a / b
        #                = a * (pi * a) / A
        #                = pi * a**2 / A
        
        # Fit an ellipse to this cross-section shape
        perim_resid = lambda a : perim - (4*a*ellipe(1 - ( A / (np.pi * a**2 ) )**2))
        if iphi == 0:
            a1 = fsolve(perim_resid, 1)
        else:
            a1 = fsolve(perim_resid, a1)
        a2 = A / (np.pi * a1)
        if a1 > a2:
            maj = a1
            min = a2
        else:
            maj = a2
            min = a1
        # Store the effective elongation
        elongs[iphi] = maj/min

    # Penalize maximum elongation
    e = np.max(elongs)
    print("Max Elongation =",e)
    print("Mean Elongation =",np.mean(elongs))
    pen = np.max([0,e-t])
    return pen

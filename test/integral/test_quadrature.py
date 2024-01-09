import numpy as np
import math 
from graphene.integral.quadrature import *
#from graphene.basis.gaussian import integral
from graphene.basis.gaussian import gaussian_integral

pi = np.pi
atol = 1e-16
    
class TestQuadratureGauss:

    npts = 50
    weights_abscissa = get_quadrature_points(npts, quadrature_type='gauss_chebyshev_2')
    
    weights = weights_abscissa[:,0]
    abscissa = weights_abscissa[:,1]
    
    a = 10.0

    def test_gauss_chebyshev_2_canonic(self):
        """testing function on canonic polynomial basis eg x^n"""
        
        for n in range(20):
            I_true = 0.0 if n%2==1 else 2.0 / (n+1)

            I_quadrature = np.dot(self.weights, self.abscissa**n)

            assert math.isclose(I_true, I_quadrature, abs_tol=atol)

    def test_gauss_chebyshev_2_gaussians(self):
        """testing on gaussians function with exponent/angular momentum over R with variable substitution on [-1,1]"""

        def angular_gaussian(x, n, a):
            return x**n * np.exp(-a*x**2.0)

        for n in range(20):

            I_true = gaussian_integral(self.a, n)
            integrand = angular_gaussian(np.arctanh(self.abscissa), n, self.a) / (1.0 - self.abscissa**2.0)
            I_quadrature = np.dot(self.weights, integrand)
            
            assert math.isclose(I_true, I_quadrature, abs_tol=atol)

class TestQuadratureLevedev:
    weights_abscissa = get_quadrature_points(None, quadrature_type='lebedev')
    weights = weights_abscissa[:,0]
    theta = weights_abscissa[:,1]
    phi = weights_abscissa[:,2]

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    def test_shapes(self):
#        assert self.weights.shape==(5810,) and self.theta.shape==(5810,) and self.phi.shape==(5810,)
#        assert self.x.shape==(5810,) and self.y.shape==(5810,) and self.z.shape==(5810,)

#        assert math.isclose(np.max(self.theta), pi, abs_tol=atol) 
#        assert math.isclose(np.min(self.theta), -pi, abs_tol=2e-2)

        assert math.isclose(np.max(self.phi), pi, abs_tol=atol) 
        assert math.isclose(np.min(self.phi), 0, abs_tol=atol)

    def test_unit_sphere(self):
        I_quad = 4*pi*np.sum(self.weights)
        assert math.isclose(4*pi, I_quad, abs_tol=atol)


    def test_2(self):
        I_quad = 4*pi*np.dot(self.weights, self.z)
        assert math.isclose(0.0, I_quad, abs_tol=4*atol)

class TestSphericalQuadrature:

    weights = R3_quadrature_points[:,0:1]
    points = R3_quadrature_points[:,1:]
    subs = subs.reshape(-1,1)

    def test_gaussian_1(self):
        """test on 3d gaussian integral"""
        R2 = np.linalg.norm(self.points, axis = 1, keepdims = True)**2
        for alpha in np.linspace(1e-1, 1e3, 10):
            I_quad = 4.0*pi*np.einsum('ij,ij', self.weights, self.subs * np.exp(-alpha*R2))
            I_true = (pi/alpha)**(3.0/2.0)
            assert np.isclose(I_true, I_quad)

    def test_gaussian_2(self):
        R2 = np.linalg.norm(self.points, axis = 1, keepdims = True)**2
        for alpha in np.linspace(1e-1, 1e3, 10):
            I_quad = 4.0*pi*np.einsum('ij,ij', self.weights,  np.prod(self.points, axis=1, keepdims=True)**2 * self.subs * np.exp(-alpha*R2))
            I_true = gaussian_integral(alpha,2) **3
            assert np.isclose(I_true, I_quad)

    def test_gaussian_3(self):
        num_R3 = len(R3_quadrature_points)
        alpha = 5.0

        # integral over t
        I_quad = 0.0
        for wkt, tk in Chebyshev_quadrature_points_01:
            R2_bar = tk
            I_tmp_R2 = 0.0

            for k2 in range(num_R3):
                wk_xyz2, x2, y2, z2 = R3_quadrature_points[k2]
                subs2 = subs[k2]
                R2 = np.array([[x2, y2, z2]]) + R2_bar # centering quadrature pts

                I_tmp_R2 += wk_xyz2 * math.exp(-alpha*np.linalg.norm(R2-R2_bar)**2) * subs2

            I_quad += wkt * 4.0 * pi * I_tmp_R2

        I = math.sqrt(pi/alpha)**3
        assert math.isclose(I_quad, I)


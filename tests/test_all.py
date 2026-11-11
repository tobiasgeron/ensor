import pytest
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


# To test:
# python -m pytest tests/test_all.py -v
#
# To get code coverage:
# python -m pytest --cov=ensor tests/test_all.py
#
# To get code coverage html report to see which lines we missed:
# python -m pytest --cov=ensor tests/test_all.py --cov-report=html
#
#
# Need to have pytest and pytest-cov installed


from ensor import linear, log, sqrt, asinh_basic, asinh_alternative, asinh, mtf, mtf_inverse
ALL_SCALING_FUNCTIONS = [linear, log, sqrt, asinh_basic, asinh_alternative, asinh, mtf, mtf_inverse]


def create_random_noise(shape = (32,32), n_bands = 3):
    imgs = []
    for band in range(n_bands):
        imgs.append(np.random.normal(0.0, 1.0, size=shape))
    return imgs





### Test Unit conversions
class TestUnitConversions:

    def test_nanomaggy_to_njy_and_back(self):
        from ensor import nanomaggy_to_njy, njy_to_nanomaggy
        assert np.isclose(nanomaggy_to_njy(1),3631)
        assert np.isclose(njy_to_nanomaggy(3631), 1)

    def test_nanomaggy_to_mag_and_back(self):
        from ensor import nanomaggy_to_mag, mag_to_nanomaggy
        
        # AB magnitude 22.5 corresponds to 1 nanomaggy by definition
        assert np.isclose(nanomaggy_to_mag(1.0), 22.5)
        assert np.isclose(mag_to_nanomaggy(22.5), 1.0)

    def test_nanomaggy_to_mag_edge_cases(self):
        from ensor import nanomaggy_to_mag
        # Zero and negative fluxes should return NaN without throwing warnings/errors
        zero_flux = np.array([0.0, -1.0])
        result = nanomaggy_to_mag(zero_flux)
        assert np.all(np.isnan(result))




class TestScalingFunctions:
    def test_linear_scaling(self):
        from ensor import linear
        x = np.array([0, 1, 2, 3])
        # Default: slope=1, intercept=0
        np.testing.assert_array_equal(linear(x), x)
        # Custom parameters
        np.testing.assert_array_equal(linear(x, slope=2, intercept=1), [1, 3, 5, 7])

    def test_sqrt_scaling(self):
        from ensor import sqrt
        x = np.array([0.0, 4.0, 9.0])
        np.testing.assert_array_equal(sqrt(x), [0.0, 2.0, 3.0])

    def test_asinh_behavior(self):
        from ensor import asinh
        x = np.array([0.0, 10.0, 100.0])
        res = asinh(x, Q=20, stretch=1, m=0)
        # Output should be non-negative due to np.maximum(val, 0)
        assert np.all(res >= 0)
        # Check monotonic increase
        assert res[2] > res[1] > res[0]



    def test_log_scaling_defaults(self):
        """Test log scaling function with default parameter a=2."""
        from ensor import log
        
        # Test x = 0: log(0 + 1) / log(3) = 0 / log(3) = 0.0
        assert np.isclose(log(0.0), 0.0)
        
        # Test x = 1: log(2*1 + 1) / log(3) = log(3) / log(3) = 1.0
        assert np.isclose(log(1.0), 1.0)

    def test_log_scaling_custom_parameter(self):
        """Test log scaling function with a custom parameter a."""
        from ensor import log
        
        # If a = 4, x = 1 -> log(4*1 + 1) / log(5) = log(5) / log(5) = 1.0
        assert np.isclose(log(1.0, a=4.0), 1.0)

    def test_asinh_basic_defaults(self):
        """Test asinh_basic with default beta=10 and x0=0."""
        from ensor import asinh_basic
        
        # At x = 0, asinh(0 / 10) = 0.0
        assert np.isclose(asinh_basic(0.0), 0.0)
        
        # At x = 10, asinh(10 / 10) = asinh(1.0)
        expected = np.arcsinh(1.0)
        assert np.isclose(asinh_basic(10.0), expected)

    def test_asinh_basic_custom_params(self):
        """Test asinh_basic with custom offset x0 and softening parameter beta."""
        from ensor import asinh_basic
        
        x = np.array([5.0, 15.0])
        # x0 = 5, beta = 5 -> (x - 5) / 5 -> [0, 2]
        result = asinh_basic(x, beta=5.0, x0=5.0)
        expected = np.asinh(np.array([0.0, 2.0]))
        
        np.testing.assert_allclose(result, expected)

    def test_asinh_alternative_defaults_and_clipping(self):
        """Test asinh_alternative defaults and verify negative values clip to 0."""
        from ensor import asinh_alternative
        
        # Test input below m (m=0) which results in negative inner values, 
        # ensuring np.maximum(val, 0) correctly clips them to 0.0
        x_negative = np.array([-5.0, 0.0, 10.0])
        result = asinh_alternative(x_negative, Q=10.0, stretch=10.0, m=0.0)
        
        assert result[0] == 0.0  # Clipped from negative
        assert result[1] == 0.0  # asinh(0) / 10 = 0.0
        assert result[2] > 0.0   # Positive scaling

    def test_asinh_alternative_custom_m_and_stretch(self):
        """Test asinh_alternative behavior with custom offset m and stretch."""
        from ensor import asinh_alternative
        
        # If x = m, (x - m) = 0, so result should be 0.0
        val_at_m = asinh_alternative(5.0, Q=10.0, stretch=5.0, m=5.0)
        assert np.isclose(val_at_m, 0.0)




class TestMTFFunctions:

    def test_mtf_identity(self):
        """Test that m=0.5 results in a linear identity mapping (f(x) = x_norm)."""
        from ensor import mtf
        
        # Values ranging from 0 to 100 (matching x_max=100)
        x = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = mtf(x, m=0.5, x_max=100.0)
        expected = x / 100.0
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mtf_boundary_conditions(self):
        """Test exact boundary behaviors: 0 maps to 0, x_max maps to 1."""
        from ensor import mtf
        
        x = np.array([0.0, 100.0])
        result = mtf(x, m=0.3, x_max=100.0)
        
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)

    def test_mtf_clipping_and_overflow(self):
        """Test that values exceeding x_max are safely clipped to 1.0."""
        from ensor import mtf
        
        x = np.array([-10.0, 50.0, 150.0]) # Below 0 and above x_max
        result = mtf(x, m=0.5, x_max=100.0)
        
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_mtf_inverse_basic_execution(self):
        """Test that mtf_inverse runs correctly on a square array and returns valid shape/bounds."""
        from ensor import mtf_inverse
        
        # Create a synthetic 20x20 square image
        np.random.seed(42)
        img = np.random.uniform(5.0, 45.0, size=(20, 20))
        
        transformed = mtf_inverse(img, alpha=0.2, crop_fraction=1.0, x_max=100.0)
        
        # Check shape preservation and output range [0, 1]
        assert transformed.shape == (20, 20)
        assert np.min(transformed) >= 0.0
        assert np.max(transformed) <= 1.0

    def test_mtf_inverse_crop_fraction_variations(self):
        """Test mtf_inverse with partial central crops (e.g., crop_fraction = 0.5)."""
        from ensor import mtf_inverse
        
        img = np.linspace(10, 80, 400).reshape(20, 20)
        
        # Should execute successfully without throwing index or slice errors
        transformed = mtf_inverse(img, alpha=0.25, crop_fraction=0.5, x_max=100.0)
        assert transformed.shape == (20, 20)

    def test_mtf_inverse_invalid_crop_fraction(self):
        """Test that invalid crop fractions outside [0, 1] trigger an AssertionError."""
        from ensor import mtf_inverse
        
        img = np.ones((10, 10))
        
        with pytest.raises(AssertionError, match="crop_fraction should be between 0 and 1"):
            mtf_inverse(img, alpha=0.2, crop_fraction=1.5, x_max=100.0)
            
        with pytest.raises(AssertionError, match="crop_fraction should be between 0 and 1"):
            mtf_inverse(img, alpha=0.2, crop_fraction=-0.1, x_max=100.0)




class TestAsinhHelpers:
    def test_calculate_Isat(self):
        from ensor import calculate_Isat
        Q = 20.0
        isat = calculate_Isat(Q)
        expected = np.sinh(np.sqrt(Q)) / Q
        assert np.isclose(isat, expected)

    def test_calculate_x_zero(self):
        from ensor import calculate_x_zero
        # x_0 = -m / (weight * stretch)
        x0 = calculate_x_zero(Q=20, weight=1.5, stretch=2.0, m=-3.0)
        assert np.isclose(x0, -(-3.0) / (1.5 * 2.0))


    def test_calculate_x_sat(self):
        from ensor import calculate_x_sat
        
        Q = 20.0
        weight = 1.5
        stretch = 2.0
        m = 0.5
        
        # Expected calculation based on derivation
        isat = np.sinh(np.sqrt(Q)) / Q
        expected_xsat = (isat - m) / (weight * stretch)
        
        result = calculate_x_sat(Q, weight, stretch, m)
        assert np.isclose(result, expected_xsat)

    def test_calculate_x_knee(self):
        from ensor import calculate_x_knee
        
        Q = 20.0
        weight = 1.5
        stretch = 2.0
        m = 0.5
        
        # Expected calculation based on derivation: x_knee = (1/Q - m) / (weight * stretch)
        expected_xknee = ((1.0 / Q) - m) / (weight * stretch)
        
        result = calculate_x_knee(Q, weight, stretch, m)
        assert np.isclose(result, expected_xknee)





class TestMakeImage:

    @pytest.mark.parametrize("func", ALL_SCALING_FUNCTIONS)
    def test_make_image_for_all_scalings(self, func):
        """Check that all scaling functions do indeed produce images"""
        from ensor import make_image

        imgs = create_random_noise()

        bands = ['g', 'r', 'i']
        weights = {
            'g': (0, 0, 1),
            'r': (0, 1, 0),
            'i': (1, 0, 0)
        }

        rgb = make_image(imgs, bands, weights=weights, func=func)

        
        


    def test_make_image_greyscale_warning(self):
        """Test that passing weights=None defaults to greyscale and raises a UserWarning."""
        from ensor import make_image, asinh
        img_g = np.random.uniform(0, 10, size=(10, 10))
        img_r = np.random.uniform(0, 10, size=(10, 10))
        
        with pytest.warns(UserWarning, match="You did not input any weights"):
            rgb = make_image([img_g, img_r], bands=['g', 'r'], weights=None, func=asinh)
            
        # Greyscale/unweighted output should have identical R, G, B channels
        assert rgb.shape == (10, 10, 3)
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
        np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])

    def test_make_image_rgb_output_shape_and_clip(self):
        """Test standard RGB composite creation and clipping bounds [0, 1]."""
        from ensor import make_image, asinh
        
        shape = (32, 32)
        imgs = [
            np.random.normal(5.0, 1.0, size=shape),
            np.random.normal(10.0, 2.0, size=shape),
            np.random.normal(20.0, 3.0, size=shape)
        ]
        bands = ['g', 'r', 'i']
        weights = {
            'g': (0, 0, 1),
            'r': (0, 1, 0),
            'i': (1, 0, 0)
        }
        
        rgb = make_image(imgs, bands, weights=weights, func=asinh, clip=True)
        
        assert rgb.shape == (32, 32, 3)
        assert np.min(rgb) >= 0.0
        assert np.max(rgb) <= 1.0

        rgb = make_image(imgs, bands, weights=weights, func=asinh, clip=True, apply_lupton = False)
                
        assert rgb.shape == (32, 32, 3)
        assert np.min(rgb) >= 0.0
        assert np.max(rgb) <= 1.0

    def test_make_image_missing_func_raises_exception(self):
        from ensor import make_image
        img = np.zeros((10, 10))
        with pytest.raises(Exception, match="You need to input a scaling/stretch function"):
            make_image([img], bands=['g'], weights={'g': (1,1,1)}, func=None)








class TestPlottingFunctions:

    def test_plot_image_basic_and_custom_args(self):
        from ensor import plot_image
        
        # Create a dummy 2D image array
        img = np.random.normal(10, 2, size=(20, 20))
        
        # Test passing an existing axis to avoid plt.show() blocking
        fig, ax = plt.subplots()
        
        # Call plot_image with custom options (colorbar, title, percentiles)
        plot_image(
            img, 
            percentile=[5.0, 95.0], 
            ax=ax, 
            cmap='viridis', 
            plot_colorbar=True, 
            title='Test Image'
        )
        
        # Verify title and ax were correctly handled
        assert ax.get_title() == 'Test Image'
        plt.close(fig)

    def test_plot_image_nan_percentile_handling(self):
        from ensor import plot_image
        
        # Image containing NaNs to verify np.nanpercentile robustness
        img = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        
        fig, ax = plt.subplots()
        # Should not raise ValueError or RuntimeWarning from NaNs
        plot_image(img, ax=ax)
        plt.close(fig)

    def test_plot_curve_standard(self):
        from ensor import plot_curve, linear
        
        fig, ax = plt.subplots()
        
        # Test plotting a standard scaling curve with nanomaggy-to-magnitude conversion
        plot_curve(
            func=linear,
            plot_yaxis_log=False,
            plot_mag=True,
            axvlines=[1.0, 5.0],
            clip=True,
            ax=ax,
            label='Linear Test',
            slope=1.0,
            intercept=0.0
        )
        
        # Check that plot generated lines and labels correctly
        assert len(ax.get_lines()) > 0
        assert ax.get_ylabel() == 'Output'
        plt.close(fig)

    def test_plot_curve_log_axis_and_no_mag(self):
        from ensor import plot_curve, sqrt
        
        fig, ax = plt.subplots()
        
        # Test with logarithmic y-axis and raw flux units (no AB magnitude conversion)
        plot_curve(
            func=sqrt,
            plot_yaxis_log=True,
            plot_mag=False,
            axvlines=[],
            clip=False,
            ax=ax,
            label='Sqrt Log Test'
        )
        
        assert ax.get_ylabel() == 'log(Output)'
        assert ax.get_xlabel() == 'Nanomaggy'
        plt.close(fig)


    def test_plot_curve_test_axes_run(self):
            from ensor import plot_curve, sqrt
            
            fig, ax = plt.subplots()
            
            # Test with both log axes
            plot_curve(
                func=sqrt,
                plot_yaxis_log=True,
                plot_mag=True,
                axvlines=[],
                clip=True,
                ax=ax
            )

            plt.close(fig)



            fig, ax = plt.subplots()
                        
            # Test with both linear axes
            plot_curve(
                func=sqrt,
                plot_yaxis_log=False,
                plot_mag=False,
                axvlines=[],
                clip=True,
                ax=ax
            )

            plt.close(fig)
    




class TestEstimateNoise:

    def test_estimate_noise_normal_distribution(self):
        from ensor import estimate_noise
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        true_sigma = 2.5
        
        # Generate a synthetic noise image with a known standard deviation
        img = np.random.normal(loc=0.0, scale=true_sigma, size=(200, 200))
        
        estimated_noise = estimate_noise(img)
        
        # The MAD estimator scaled by 1.4826 should closely approximate the true sigma
        assert np.isclose(estimated_noise, true_sigma, rtol=0.1)

    def test_estimate_noise_with_nans(self):
        from ensor import estimate_noise
        
        np.random.seed(42)
        img = np.random.normal(loc=0.0, scale=1.5, size=(100, 100))
        
        # Inject NaNs to test nanmedian robustness
        img[20:30, 20:30] = np.nan
        
        estimated_noise = estimate_noise(img)
        
        # Should complete without error, ignore NaNs, and give a valid positive noise estimate
        assert not np.isnan(estimated_noise)
        assert estimated_noise > 0.0

    def test_estimate_noise_constant_image(self):
        from ensor import estimate_noise
        
        # A uniform flat image has no background variation/noise
        img = np.ones((50, 50)) * 10.0
        estimated_noise = estimate_noise(img)
        
        assert np.isclose(estimated_noise, 0.0)
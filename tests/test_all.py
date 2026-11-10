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
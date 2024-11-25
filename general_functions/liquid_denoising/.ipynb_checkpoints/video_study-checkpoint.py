'''
Functions used to analyze the evolution and growth of nanoparticles
'''
from math import erfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from ..general_functions import validate_array, format_uncertainty, print_uncertainty
from .general_classes import Frames, Calibration, FittedEllipses

PI = np.pi

class VideoStudied:
    '''A class representing the video we are working with.
    '''
    def __init__(self, name, frames, calibration=Calibration(), tracked_particles=None):
        '''Initializes a VideoStudied instance.
        
        Parameters
        ----------
        name : str
            Name of the video. Used as the base for naming all of the video related informations.
        frames : Frames
            Frames related information about the video.
        calibration : Calibration
            Calibration used in the video to go from pixels to the unit of interest.
        '''
        self.name = name
        self.frames = frames
        self.calibration = calibration

        if tracked_particles is None:
            self.tracked_particles = []
            self.tracked_particles_names = []
        else:
            self.add_particles(tracked_particles)

    def __repr__(self):
        return f"Video {self.name} with {len(self.tracked_particles)} tracked particle(s)"

    def add_particles(self, tracked_particles):
        '''Add some TrackedParticle instances to the current VideoStudied.
        
        Parameters
        ----------
        tracked_particles : TrackedParticle or list of TrackedParticle
            Particles in the Video
        '''
        if isinstance(tracked_particles, TrackedParticle):
            tracked_particles = [tracked_particles, ]

        if isinstance(tracked_particles, (list, tuple)):
            for particle in tracked_particles:
                if isinstance(particle, TrackedParticle):
                    if particle.name in self.tracked_particles_names:
                        print(f"{particle.name} was not added to the video because it has"
                               " the same name as one of the particles already tracked")
                    else:
                        self.tracked_particles.append(particle)
                        self.tracked_particles_names.append(particle.name)
                else:
                    raise TypeError("The particles added to the video must be TrackedParticle")

    # def __item_save(self, file, key, value):
    #     print(key, value)
    #     if isinstance(value, (Frames, Calibration, FittedEllipses, TrackedParticle, Tracking)):
    #         inner_group = file.create_group(key)
    #         for inner_key, inner_value in value.__dict__.items():
    #             self.__item_save(inner_group, inner_key, inner_value)
    #     else:
    #         file[key] = value

    # def save(self, file_path):
    #     '''Save the current VideoStudied instance into a single file.

    #     Parameters
    #     ----------
    #     file_path : str
    #         Path to the saved file
    #     '''
    #     with h5py.File(file_path, "w") as file:
    #         for key, value in self.__dict__.items():
    #             self.__item_save(file, key, value)

    #     with h5py.File(file_path, 'w') as file:
    #         for key, value in self.__dict__.items():
    #             if isinstance(value, TrackedParticle):
    #                 inner_group = file.create_group(key)
    #                 for inner_key, inner_value in value.__dict__.items():
    #                     inner_group[inner_key] = inner_value
    #             else:
    #                 file[key] = value



class TrackedParticle:
    '''Define all the informations linked to the tracked particle

    '''
    def __init__(self, name, frames, particle_morpho, tracking=None, calibration=Calibration()):
        '''Initialize a Tracked Particle instance.
        
        Parameters
        ----------
        name : str
            Name of the track. (e.g. "Track#01")
        frames : Frames
            Frames on which we study this tracked particle.
        particle_morpho : str
            Morphology of the tracked particle.
        tracking : Tracking
            Tracking of the TrackedParticle using a specific method.
        calibration : Calibration
            Link between the size in pixel and the one in the unit of interest.
        '''
        self.name = name
        self.frames = frames

        if not particle_morpho in ["NanoCube", "NanoRod", "NanoSphere"]:
            self.particle_morpho = None
            print("The morphology selected was not implemented yet, try 'NanoCube', 'NanoRod' or"
                  " 'NanoSphere'")
        else:
            self.particle_morpho = particle_morpho

        if tracking is None:
            self.tracking = []
            self.tracking_names = []
        else:
            self.add_tracking(tracking)

        self.calibration = calibration

    def __repr__(self):
        if self.particle_morpho is None :
            return (f"Tracked Particle {self.name} has {len(self.tracking)}"
                    " differents tracking")
        return (f"Tracked Particle {self.name} is a {self.particle_morpho} and has"
                f" {len(self.tracking)} differents tracking")

    def add_tracking(self, tracking):
        '''Add some Tracking instances to the current TrackedParticle.
        
        Parameters
        ----------
        tracking : Tracking or list of Tracking
            Tracking associated to the TrackedParticle
        '''
        if isinstance(tracking, Tracking):
            tracking = [tracking, ]

        if isinstance(tracking, (list, tuple)):
            for method in tracking:
                if isinstance(method, Tracking):
                    if method.name in self.tracking_names:
                        print(f"{method.name} was not added to the tracked particles because it has"
                               " the same name as one of the tracking already existing")
                    else:
                        self.tracking.append(method)
                        self.tracking_names.append(method.name)
                else:
                    raise TypeError("The trackings added to the tracked particles must be Tracking")

    def plot_area(self):
        '''Plot all the evolution of the area of the tracked particle for all trackings
        '''
        print("*Not implemented yet")



class Tracking:
    '''A class containing all informations related the result of the tracking of a particle
    '''
    def __init__(self, name, area=None, time=None, particle_morpho=None,
                 fitted_ellipses=FittedEllipses(), calibration=Calibration()):
        '''Initialize a Tracking instance.

        Parameters:
        -----------
        name : str
            Name of the tracking. (e.g. "6.2-Hand-Adrien")
        area : array-like
            Array of the area of the TrackedParticle at each frame.
        time : array-like
            Array of the time in second at which each point was measure.
        particle_morpho : str
            Morphology of the particle in the tracking.
            Are implemented only NanoCube and NanoRod.
        calibration : Calibration
            Link between the pixel and the unit of interest
        '''
        self.name = name

        if area is None:
            area = []
        if time is None:
            time = []

        area = validate_array(area, "area")
        if np.any(area!=np.abs(area)):
            raise ValueError("area must always be positive")

        time = validate_array(time, "time")
        if len(area)!=len(time):
            raise ValueError("area and time must have the same length")

        self._area = area
        self.frames = Frames(len(time),1)
        self.frames.time = np.array(time)

        if particle_morpho not in ("NanoCube", "NanoRod", "NanoSphere"):
            self._particle_morpho = None
            print("The morphology selected was not implemented yet, try 'NanoCube', 'NanoRod', or"
                  " 'NanoSphere'")
        else:
            self._particle_morpho = particle_morpho
            if not isinstance(fitted_ellipses, FittedEllipses):
                self._fitted_ellipses = None
            elif fitted_ellipses.size!=len(area):
                self._fitted_ellipses = None
            else:
                self._fitted_ellipses = fitted_ellipses

        if not isinstance(calibration, Calibration):
            raise TypeError("calibration must be a Calibration object")
        self._calibration = calibration

        self._u_typ_area = ["percentage", 10]
        self._u_typ_time = ["fixed", self.frames.frame_duration]
        # self._fit_params = None

    def __repr__(self):
        if self.particle_morpho is None :
            return (f"Tracking '{self.name}' of some particle on {self.frames.number_of_frames}"
                    " points")
        return (f"Tracking '{self.name}' of a {self.particle_morpho} on"
                f" {self.frames.number_of_frames} points")

    @property
    def area(self):
        '''Returns the array of area measured for this tracking.
        
        Returns
        -------
        out : numpy.ndarray
            Array of area in pixels squared.
        '''
        return self._area

    @area.setter
    def area(self, array):
        '''Defines the area for each frames during the tracking.

        Parameters
        ----------
        array : array-like
            New array of area.
        '''
        array = validate_array(array, "area")
        if np.any(array!=np.abs(array)):
            raise ValueError("area must always be positive")
        if len(array)!=self.frames.number_of_frames:
            raise ValueError(f"area must have {self.frames.number_of_frames} elements like time")
        self._area = array

    @property
    def u_type_area(self):
        '''Returns the type and value of the uncertainty for the area in square pixels.

        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value
        '''
        return self._u_typ_area

    def set_u_type_area(self, value, fixed=True):
        '''Defines the type and value of the uncertainty associated to the area in square pixels.

        Parameters
        ----------
        value : int, float
            Value of the uncertainty to apply.
        fixed : bool
            Defines if the uncertainty is fixed or a percentage of the measurement considered
        '''
        if fixed:
            self._u_typ_area = ["fixed", value]
        else:
            self._u_typ_area = ["percentage", abs(value)]

    @property
    def u_area(self):
        '''Returns the array of uncertainty of the area in square pixels as defined by the
         u_area_rate.
         
        Returns
        -------
        out : numpy.ndarray
            Array of area uncertainty.
        '''
        if self._u_typ_area[0]=="fixed":
            return np.ones_like(self.area) * self._u_typ_area[1]
        return self.area * self._u_typ_area[1] / 100

    @property
    def time(self):
        '''Returns the array of the time associated to the tracking.
        
        Returns
        -------
        out : numpy.ndarray
            Array of time in seconds.
        '''
        return self.frames.time

    @time.setter
    def time(self, array):
        '''Defines the time for each area measured during the tracking.

        Parameters
        ----------
        array : array-like
            New array of time.
        '''
        if not (isinstance(array, (list, tuple, np.ndarray)) and
                (all(isinstance(value, (int, float)) for value in array))):
            raise TypeError("time must be an array of integers or floating-point numbers")
        if len(array)!=len(self.area):
            raise ValueError(f"time must have {len(self.area)} elements like area")
        self.frames.time = np.array(array)

    @property
    def u_type_time(self):
        '''Returns the type and value of the uncertainty for the time in seconds.

        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value
        '''
        return self._u_typ_time

    def set_u_type_time(self, value, fixed=True):
        '''Defines the type and value of the uncertainty associated to the time in seconds.

        Parameters
        ----------
        value : int, float
            Value of the uncertainty to apply.
        fixed : bool
            Defines if the uncertainty is fixed or a percentage of the measurement considered
        '''
        if fixed:
            self._u_typ_time = ["fixed", value]
        else:
            self._u_typ_time = ["percentage", abs(value)]

    @property
    def u_time(self):
        '''Returns the array of uncertainty of the time in seconds as defined by the u_time_rate.
         
        Returns
        -------
        out : numpy.ndarray
            Array of time uncertainty.
        '''
        if self._u_typ_time[0]=="fixed":
            return np.ones_like(self.time) * self._u_typ_time[1]
        return self.time * self._u_typ_time[1] / 100

    @property
    def particle_morpho(self):
        '''Returns the morphology of the particle.
        
        Returns
        -------
        out : str
            Morphology of the particle.
        '''
        return self._particle_morpho

    @particle_morpho.setter
    def particle_morpho(self, morpho_name):
        '''Defines the morphology of the particle.
        
        Parameters
        ----------
        morpho_name : str
            Morphology of the particle.
        '''
        if morpho_name not in ("NanoCube", "NanoRod", "NanoSphere"):
            self._particle_morpho = None
            print("The morphology selected was not implemented yet, try 'NanoCube', 'NanoRod', or"
                " 'NanoSphere'")
        else:
            self._particle_morpho = morpho_name

    @property
    def fitted_ellipses(self):
        '''Return the FittedEllipses associated to each Tracking frames.
        
        Returns
        -------
        out : FittedEllipses
            FittedEllipses object.
        '''
        return self._fitted_ellipses

    @fitted_ellipses.setter
    def fitted_ellipses(self, new_ellipses):
        '''Defines the new ellipses fitted on the particle at each frames of the Tracking.

        Parameters
        ----------
        new_ellipses : FittedEllipses
            New FittedEllipses object to use from now on.
        '''
        if not isinstance(new_ellipses, FittedEllipses):
            raise TypeError("new_ellipses must be a FittedEllipses object")
        if new_ellipses.size != self.frames.number_of_frames:
            raise ValueError("The size of new_ellipses must be the same as the number of frames")
        self._fitted_ellipses = new_ellipses

    @property
    def calibration(self):
        '''Returns the calibration of the video, used for the tracking.
        
        Returns
        -------
        out : Calibration
            Calibration used to change the units.
        '''
        return self._calibration

    @calibration.setter
    def calibration(self, new_calib):
        '''Defines the new calibration to use for the calculations.
        
        Parameters
        ----------
        new_calib : Calibration
            New calibration object that will be use from now on.
        '''
        if not isinstance(new_calib, Calibration):
            raise TypeError("new_calib must be a Calibration object")
        self._calibration = new_calib

    # @property
    # def fit_params(self):
    #     '''Returns the fit parameters if they are calculated.

    #     Returns
    #     -------
    #     out : list
    #         List of the fitted parameters in this order : (a, b, u_a, u_b).
    #     '''
    #     if self._fit_params is None:
    #         raise ValueError("You must perform a get_volume_growth, get_unit_volume_growth or"
    #                          " get_absorption_rate before hand.")
    #     return self._fit_params

    @property
    def unit_area(self):
        '''Area in square calibration units.

        Returns
        -------
        out : numpy.ndarray
            Array of area in square calibrated units.
        '''
        return self.area * self.calibration.unit_per_px**2

    @property
    def u_unit_area(self):
        '''Uncertainty of the area in square calibration units.
        
        Returns
        -------
        out : numpy.ndarray
            Array of the area uncertainty.
        '''
        return self.u_area * self.calibration.unit_per_px**2

    @property
    def est_volume(self):
        '''Estimates the volume associated to the area and the particle morphology.
        
        Returns
        -------
        out : numpy.ndarray
            Array of estimated volume in cube pixel.
        '''
        if self.particle_morpho == "NanoCube":
            return self.area**1.5
        if self.particle_morpho == "NanoRod":
            return self.area * self.fitted_ellipses.minor_length
        if self.particle_morpho == "NanoSphere":
            return self.area**1.5 * (4/(3 * PI**0.5))
        print("The volume can't be estimated because the morphology selected was not implemented")
        return np.array([None]*self.frames.number_of_frames)

    @property
    def u_est_volume(self):
        '''Uncertainty on the estimated volume.
        
        Returns
        -------
        out : numpy.ndarray
            Array of the uncertainty of the estimated volume.
        '''
        if self.particle_morpho == "NanoCube":
            return (3/2) * self.area**0.5 * self.u_area
        if self.particle_morpho == "NanoRod":
            temp = ((self.u_area/self.area)**2 +
                    (self.fitted_ellipses.u_minor_length/self.fitted_ellipses.minor_length)**2)**0.5
            return self.est_volume * temp
        if self.particle_morpho == "NanoSphere":
            return 2 * (self.area/PI)**0.5 * self.u_area
        print("The volume can't be estimated because the morphology selected was not implemented")
        return np.array([None]*self.frames.number_of_frames)

    @property
    def unit_est_volume(self):
        '''Estimated volume in cube calibration units depending on the area and particle morphology.

        Returns
        -------
        out : numpy.ndarray
            Array of volume in cube calibrated units.
        '''
        return self.est_volume * self.calibration.unit_per_px**3

    @property
    def u_unit_est_volume(self):
        '''Uncertainty on the estimated volume.
        
        Returns
        -------
        out : numpy.ndarray
            Array of the uncertainty of the estimated volume.
        '''
        return self.u_est_volume * self.calibration.unit_per_px**3

    def update_particle(self, tracked_particle, subframes_indices, time_shift=0,
                        frames_pov="local"):
        '''Update informations of the tracking based on informations of the tracked particle.

        Parameters
        ----------
        tracked_particle : TrackedParticle
            Tracked particule followed by the tracking.
        subframes_indices : array-like
            Subset of the indices of frames on which the tracking is performed.
        time_shift : float
            Time shift to apply to all other points than the first, to consider the error of in
            time.
        frames_pov : str
            Define the frame out of which the indices are considered.
            For more details check the Frame.subframe() method.
        '''
        if len(subframes_indices)!=self.frames.number_of_frames:
            raise ValueError("subframes_indices must have the same length as the area defined")
        self.frames = tracked_particle.frames.subframes(subframes_indices, time_shift=time_shift,
                                                        frames_pov=frames_pov)
        self.particle_morpho = tracked_particle.particle_morpho
        self.calibration = tracked_particle.calibration

    def import_measurements_df(self, dataframe, tracked_particle=None, center_of_part=None,
                               radius=7, time_shift=0):
        '''Import area measurements from a pandas DataFrame based on a tracking method.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame generated by ImageJ or from any software for data processing.
        tracked_particle : TrackedParticle
            Particle to associate to the current tracking.
        center_of_mass : array-like
            Coordinates of the center_of_mass of the considered particle
        timeshift : float
            Added time added only to the first frame of the tracking
        '''
        if center_of_part is None:
            indices = dataframe.index.to_numpy()
        elif isinstance(center_of_part, (list, tuple, np.ndarray)) and len(center_of_part)==2:
            x = dataframe["X"].to_numpy()
            y = dataframe["Y"].to_numpy()
            indices = np.where((x>(center_of_part[0]-radius)) & (x<(center_of_part[0]+radius)) &
                                (y>(center_of_part[1]-radius)) & (y<(center_of_part[1]+radius)))
            indices = dataframe.index.to_numpy()[indices]
        else:
            raise TypeError("center_of_part must be an array-like or a None")

        if tracked_particle is None:
            self.frames = Frames(len(indices),1)
            self.area = dataframe["Area"].loc[indices].to_numpy()
            if "Time" in dataframe.columns:
                self.frames.time = np.array(dataframe["Time"].loc[indices].to_numpy())
            elif "Slice" in dataframe.columns:
                self.frames.time = np.array(dataframe["Slice"].loc[indices].to_numpy())
            else:
                self.frames.time = np.array(indices)
        elif isinstance(tracked_particle, TrackedParticle):
            self.name = tracked_particle.name + " - " + self.name
            if "Slice" in dataframe.columns:
                slice_indices = dataframe["Slice"].loc[indices].to_numpy()
                self.frames = Frames(len(slice_indices),1)
                self.area = dataframe["Area"].loc[indices].to_numpy()
                self.update_particle(tracked_particle, slice_indices, time_shift=time_shift)
            else:
                self.frames = Frames(len(indices),1)
                self.area = dataframe["Area"].loc[indices].to_numpy()
                self.update_particle(tracked_particle, indices, time_shift=time_shift)

        if "Major" in dataframe.columns and "Minor" in dataframe.columns:
            major = dataframe["Major"].loc[indices].to_numpy()
            minor = dataframe["Minor"].loc[indices].to_numpy()
            self._fitted_ellipses = FittedEllipses(major_length=major, minor_length=minor)
        else:
            self._fitted_ellipses = FittedEllipses()

    def plot_area(self, ax=None, fit=False, min_time=0, max_time=None, color="C0", color_fit="C1",
                  label=None):
        '''Plot the area as a function of time for this Tracking.
        '''
        new = False
        if ax is None:
            new = True
            fig, ax = plt.subplots()
        ax.errorbar(self.time, self.area, self.u_area, self.u_time, ls="", color=color,
                    label=label)
        ax.set_title(self.name + "\nEvolution of the area during the growth")
        if fit:
            prm, hull = self.get_area_growth(min_time=min_time, max_time=max_time, hull=True)
            ax.plot(hull[0], prm[0]* 10**prm[4] * hull[0] + prm[1] * 10**prm[5] , color=color_fit)
            ax.fill_between(hull[0], hull[1], hull[2], color=color_fit, alpha=.2)
            ax.set_title(self.name + "\nEvolution of the area during the growth\n"
                         f"{print_uncertainty(prm[0], prm[2], prm[4])} * x + "
                         f"{print_uncertainty(prm[1], prm[3], prm[5])}")
        ax.set_xlabel("Time (in s)")
        ax.set_ylabel("Area of the nanoparticle projection (in px2)")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([min(hull[0][0]-1, 0), xlim[1]])
        ax.set_ylim([min (np.min(hull)-1, 0), ylim[1]])

        if new:
            return fig, ax
        return ax

    def plot_unit_area(self, ax=None, fit=False, min_time=0, max_time=None,
                       color="C0", color_fit="C1", label=""):
        '''[*]Plot the area as a function of time for this Tracking.
        '''
        new = False
        if ax is None:
            new = True
            fig, ax = plt.subplots()
        ax.errorbar(self.time, self.unit_area, self.u_unit_area, self.u_time, ls="", color=color,
                    label=label)
        ax.set_title(self.name + "\nEvolution of the area during the growth")
        if fit:
            prm, hull = self.get_unit_area_growth(min_time=min_time, max_time=max_time, hull=True)
            ax.plot(hull[0], prm[0]* 10**prm[4] * hull[0] + prm[1] * 10**prm[5] , color=color_fit)
            ax.fill_between(hull[0], hull[1], hull[2], color=color_fit, alpha=.2)
            ax.set_title(self.name + "\nEvolution of the area during the growth\n"
                         f"{print_uncertainty(prm[0], prm[2], prm[4])} * x + "
                         f"{print_uncertainty(prm[1], prm[3], prm[5])}")
        ax.set_xlabel("Time (in s)")
        ax.set_ylabel("Area of the nanoparticle projection (in nm2)")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([min(hull[0][0]-1, 0), xlim[1]])
        ax.set_ylim([min (np.min(hull)-1, 0), ylim[1]])
        if new:
            return fig, ax
        return ax

    def plot_est_volume(self, ax=None, fit=False, min_time=0, max_time=None,
                        color="C0", color_fit="C1", label=None):
        '''[*]Plot the area as a function of time for this Tracking.
        '''
        new = False
        if ax is None:
            new = True
            fig, ax = plt.subplots()
        ax.errorbar(self.time, self.est_volume, self.u_est_volume, self.u_time, ls="", color=color,
                    label=label)
        ax.set_title(self.name + "\nEvolution of the area during the growth")
        if fit:
            prm, hull = self.get_volume_growth(min_time=min_time, max_time=max_time, hull=True)
            ax.plot(hull[0], prm[0]* 10**prm[4] * hull[0] + prm[1] * 10**prm[5] , color=color_fit)
            ax.fill_between(hull[0], hull[1], hull[2], color=color_fit, alpha=.2)
            ax.set_title(self.name + "\nEvolution of the volume during the growth\n"
                         f"{print_uncertainty(prm[0], prm[2], prm[4])} * x + "
                         f"{print_uncertainty(prm[1], prm[3], prm[5])}")
        ax.set_xlabel("Time (in s)")
        ax.set_ylabel("Estimation of the volume of the nanoparticle (in px3)")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([min(hull[0][0]-1, 0), xlim[1]])
        ax.set_ylim([min (np.min(hull)-1, 0), ylim[1]])

        if new:
            return fig, ax
        return ax

    def plot_unit_est_volume(self, ax=None, fit=False, min_time=0, max_time=None,
                             color="C0", color_fit="C1", label=None):
        '''[*]Plot the area as a function of time for this Tracking.
        '''
        new = False
        if ax is None:
            new = True
            fig, ax = plt.subplots()
        ax.errorbar(self.time, self.unit_est_volume, self.u_unit_est_volume,
                    self.u_time, ls="", marker="o", color=color, label=label)
        ax.set_title(self.name + "\nEvolution of the area during the growth")
        if fit:
            prm, hull = self.get_unit_volume_growth(min_time=min_time, max_time=max_time, hull=True)
            ax.plot(hull[0], prm[0]* 10**prm[4] * hull[0] + prm[1] * 10**prm[5] , color=color_fit)
            ax.fill_between(hull[0], hull[1], hull[2], color=color_fit, alpha=.2)
            ax.set_title(self.name + "\nEvolution of the volume during the growth\n"
                         f"{print_uncertainty(prm[0], prm[2], prm[4])} * x + "
                         f"{print_uncertainty(prm[1], prm[3], prm[5])}")
        ax.set_xlabel("Time (in s)")
        ax.set_ylabel("Estimation of the volume of the nanoparticle (in nm3)")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([min(hull[0][0]-1, 0), xlim[1]])
        ax.set_ylim([min (np.min(hull)-1, 0), ylim[1]])

        if new:
            return fig, ax
        return ax

    def plot_absorbtion_rate(self, density, ax=None, fit=False, min_time=0, max_time=None,
                        color="C0", color_fit="C1", label=None):
        '''[*]Plot the area as a function of time for this Tracking.
        '''
        new = False
        if ax is None:
            new = True
            fig, ax = plt.subplots()
        ax.errorbar(self.time, self.unit_est_volume * density, self.u_unit_est_volume * density,
                    self.u_time, ls="", color=color, label=label)
        ax.set_title(self.name + "\nEvolution of the area during the growth")
        if fit:
            prm, hull = self.get_absorption_rate(density, min_time=min_time,
                                                 max_time=max_time, hull=True)
            ax.plot(hull[0], prm[0]* 10**prm[4] * hull[0] + prm[1] * 10**prm[5] , color=color_fit)
            ax.fill_between(hull[0], hull[1], hull[2], color=color_fit, alpha=.2)
            ax.set_title(self.name + "\nEvolution of the number of atoms during the growth\n"
                         f"{print_uncertainty(prm[0], prm[2], prm[4])} * x + "
                         f"{print_uncertainty(prm[1], prm[3], prm[5])}")
        ax.set_xlabel("Time (in s)")
        ax.set_ylabel("Estimation of the number of atoms in the nanoparticle (in atoms)")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([min(hull[0][0]-1, 0), xlim[1]])
        ax.set_ylim([min (np.min(hull)-1, 0), ylim[1]])

        if new:
            return fig, ax
        return ax

    def __lineq_err(self, p, x, y, dx, dy):
        return (y - p[0]*x - p[1]) / np.sqrt((p[0]*dx)**2 + dy**2 + 1e-8)

    def __fit(self, x, y, dx, dy, hull=False):
        '''[*] fitutils thanks
        '''
        cdf_min = 0.5 * erfc(1 / 2**0.5) # CDF normal distribution P(X<=-sigma)
        cdf_max = 1 - cdf_min # CDF normal distribution P(X<=sigma)

        a0 = np.mean((y[1:]-y[:-1]) / (x[1:]-x[:-1]))
        p0 = np.array([a0, np.mean(y - (a0 * x))])
        re = np.zeros((1000, 2))
        rd = np.random.default_rng()
        for i in range(1000):
            xi = rd.normal(x, dx)
            yi = rd.normal(y, dy)
            ls = least_squares(self.__lineq_err, p0, args=(xi, yi, dx, dy))
            re[i] = ls.x
        p = np.median(re, axis=0)
        s_l = np.abs(np.quantile(re, cdf_min, axis=0) - p)
        s_h = np.abs(np.quantile(re, cdf_max, axis=0) - p)
        s = np.mean([s_l, s_h], axis=0)
        a, u_a, e_a = format_uncertainty(p[0], s[0])
        b, u_b, e_b = format_uncertainty(p[1], s[1])

        if hull:
            xl = np.min([(x[1]-x[0])/2, (x[-1]-x[-2])/2])
            x_min = x[0] - xl
            x_max = x[-1] + xl
            xplot = np.linspace(x_min, x_max, 1000)
            xparr = xplot * np.ones((1000, 1000))
            mesha = np.meshgrid(np.ones(1000), re[:, 0])[1]
            meshb = np.meshgrid(np.ones(1000), re[:, 1])[1]
            alldr = xparr * mesha + meshb
            hull_l = np.quantile(alldr, cdf_min, axis=0)
            hull_h = np.quantile(alldr, cdf_max, axis=0)
            return (np.array([a, b, u_a, u_b, e_a, e_b]), np.array([xplot, hull_l, hull_h]))

        return np.array([a, b, u_a, u_b, e_a, e_b])

    def get_area_growth (self, min_time=0, max_time=None, hull=False):
        '''[*] Fit the estimated volume points between min_time and max_time to get the rate of
         growth of the volume of the nanoparticle.
        '''
        if max_time is None:
            max_time = 1e8
        lim_time = np.where((self.time >= min_time) & (self.time <= max_time))
        if len(self.time[lim_time])<2:
            raise ValueError("The time window selected is two small and there are not enough"
                             f"points ({len(self.time[lim_time])} point)")
        x = self.time[lim_time]
        y = self.area[lim_time]
        dx = self.u_time[lim_time]
        dy = self.u_area[lim_time]
        return self.__fit(x, y, dx, dy, hull=hull)

    def get_unit_area_growth (self, min_time=0, max_time=None, hull=False):
        '''[*] Fit the estimated volume points between min_time and max_time to get the rate of
         growth of the volume of the nanoparticle.
        '''
        if max_time is None:
            max_time = 1e8
        lim_time = np.where((self.time >= min_time) & (self.time <= max_time))
        if len(self.time[lim_time])<2:
            raise ValueError("The time window selected is two small and there are not enough"
                             f"points ({len(self.time[lim_time])} point)")
        x = self.time[lim_time]
        y = self.unit_area[lim_time]
        dx = self.u_time[lim_time]
        dy = self.u_unit_area[lim_time]
        return self.__fit(x, y, dx, dy, hull=hull)

    def get_volume_growth (self, min_time=0, max_time=None, hull=False):
        '''[*] Fit the estimated volume points between min_time and max_time to get the rate of
         growth of the volume of the nanoparticle.
        '''
        if max_time is None:
            max_time = 1e8
        lim_time = np.where((self.time >= min_time) & (self.time <= max_time))
        if len(self.time[lim_time])<2:
            raise ValueError("The time window selected is two small and there are not enough"
                             f"points ({len(self.time[lim_time])} point)")
        x = self.time[lim_time]
        y = self.est_volume[lim_time]
        dx = self.u_time[lim_time]
        dy = self.u_est_volume[lim_time]
        return self.__fit(x, y, dx, dy, hull=hull)

    def get_unit_volume_growth (self, min_time=0, max_time=None, hull=False):
        '''[*] Fit the estimated volume points between min_time and max_time to get the rate of
         growth of the volume of the nanoparticle.
        '''
        if max_time is None:
            max_time = 1e8
        lim_time = np.where((self.time >= min_time) & (self.time <= max_time))
        if len(self.time[lim_time])<2:
            raise ValueError("The time window selected is two small and there are not enough"
                             f"points ({len(self.time[lim_time])} point)")
        x = self.time[lim_time]
        y = self.unit_est_volume[lim_time]
        dx = self.u_time[lim_time]
        dy = self.u_unit_est_volume[lim_time]
        return self.__fit(x, y, dx, dy, hull=hull)

    def get_absorption_rate(self, density, min_time=0, max_time=None, hull=False):
        '''[*] Fit the estimated volume points between min_time and max_time to get the rate of
         growth of the volume of the nanoparticle.
        '''
        if max_time is None:
            max_time = 1e8
        lim_time = np.where((self.time >= min_time) & (self.time <= max_time))
        if len(self.time[lim_time])<2:
            raise ValueError("The time window selected is two small and there are not enough"
                             f"points ({len(self.time[lim_time])} point)")
        x = self.time[lim_time]
        y = self.unit_est_volume[lim_time] * density
        dx = self.u_time[lim_time]
        dy = self.u_unit_est_volume[lim_time] * density
        return self.__fit(x, y, dx, dy, hull=hull)

        # if tracked_particle is None:


        #     if not (isinstance(area, (list, tuple, np.ndarray)) and
        #             (all(isinstance(value, (int, float)) for value in area))):
        #         raise TypeError("area must be an array of integers or floating-point numbers")
        # else:



        # if not ((isinstance(area, (list, tuple)) and
        #          all(isinstance(x, (int, float)) for x in area)) or
        #         isinstance(area, np.ndarray) and (area.dtype==int or area.dtype==float)):
        #     raise TypeError("Area must be an array of integers or floating-point numbers")
        # self.area = np.array(area, dtype=float)
        # self.frames = self.frames.subframes(frames_indices)

'''
Functions used to analyze the evolution and growth of nanoparticles
'''
import numpy as np

# import h5py

from ..general_functions import validate_array

PI = np.pi

class Frames:
    '''A class containing all of the frames and associated time.
    '''

    def __init__(self, number_of_frames, frame_duration=1):
        '''Defines a set of frames with associated time.
        
        Parameters:
        -----------
        number_of_frames : int
            Total number of frames.
        frame_duration : float
            Duration of each frames in seconds.
        '''
        if number_of_frames is None:
            number_of_frames = 0

        self.number_of_frames = number_of_frames
        self.frame_duration = frame_duration
        self._global_frames = np.arange(number_of_frames) + 1
        self._local_frames = np.arange(number_of_frames) + 1
        self._time = np.arange(0, self.number_of_frames * self.frame_duration, self.frame_duration)

    @property
    def loc_frames(self):
        '''Reader of the array of the frames available (from the local point of view)
        
        Returns
        -------
        out : numpy.ndarray
            Array of frames
        '''
        return self._local_frames

    @property
    def frames(self):
        '''Reader of the array of the frames available (from the global point of view of the video)
        
        Returns
        -------
        out : numpy.ndarray
            Array of frames
        '''
        return self._global_frames

    @frames.setter
    def frames(self, array):
        '''        Setter of the array of the frames available
        The new array must have has many elements as the total number of frames
        
        Parameters
        ----------
        array : numpy.ndarray
            New array of frames
        '''
        if isinstance(array, np.ndarray) and self.number_of_frames==len(array):
            self._global_frames = array
        else:
            raise ValueError("To change the value of the frames, please input an array")

    @property
    def time(self):
        '''        Reader of the array of the time associated to the frames.
        
        Returns
        -------
        out : numpy.ndarray
            Array of time.
        '''
        return self._time

    @time.setter
    def time(self, array):
        '''        Setter of the array of the time associated to the frames.
        The new array must have has many elements as the total number of frames.
        
        Parameters
        ----------
        array : numpy.ndarray
            New array to use.
        '''
        if isinstance(array, np.ndarray) and self.number_of_frames==len(array):
            self._time = array
        else:
            raise ValueError("To change the value of the time, please input an array")

    def subframes(self, frames_indices, time_shift=0, frames_pov="local"):
        '''Create a sub Frames instance for tracking particles.

        Parameters
        ----------
        frames_indices : list of int
            Indices of the frames we want to keep in the sub frames.
        time_shift : float
            Time shift to apply to all other points than the first, to consider the error of in
            time.
        frames_pov : str
            Define the frame of which element we consider.
            -'local' means that we consider the indices of the frames from 1 to number of frames
            included.
            -'global' means that we consider the indices of the frames from the previous extraction.
            In a initial video with 5 frames [1,2,3,4,5], the global and local are the same.
            We extract a first set of frames [1,3,5] from the video.
            Here global is [1,3,5] and local will be [1,2,3].
            This local renormalization is useful when considering the tracking of the particles in
            subvideos.
        '''
        if ((isinstance(frames_indices, (list, tuple)) and all(isinstance(x, int)
                                                          for x in frames_indices)) or
            (isinstance(frames_indices, np.ndarray) and frames_indices.dtype in [int, np.int64])):

            if frames_pov=="global":
                self_frames = self.frames
            elif frames_pov=="local":
                self_frames = self.loc_frames
            else:
                raise ValueError("frame_pov selected not implemented")

            if not all(x in self_frames for x in frames_indices):
                raise ValueError("frame_indices must contains indices from the current Frames")
            frames_id = np.array([np.where(x==self_frames) for x in frames_indices]).flatten()
            if len(frames_id)!=len(frames_indices):
                raise ValueError("There was some kind of problems we the indices of frames_indices")
            sub_frame = Frames(len(frames_id), frame_duration=self.frame_duration)
            sub_frame.frames = self.frames[frames_id]
            sub_frame.time = (self.time[frames_id] +
                              ([0] + [time_shift]*(len(frames_id)-1)))

        else:
            raise TypeError("frame_indices must be a list of indices")

        return sub_frame



class Calibration:
    '''Defines the spatial calibration associated with a video.
    '''
    def __init__(self, unit_per_px=1, px_per_unit=None, unit="px"):
        '''Create a calibration instance.        
        
        Parameters:
        -----------
        unit_per_px : int, float
            Number of units in a single pixel.
            If None, px_per_unit will be used.
        px_per_unit : int, float
            Number of pixels in a single unit.
            Will be used if unit_per_px is None.
        unit : str
            Name of the unit used in the calibration
        '''
        if isinstance(unit_per_px, (int, float)):
            self._unit_per_px = float(unit_per_px)
            self._px_per_unit = 1 / float(unit_per_px)
        elif unit_per_px is None:
            if isinstance(px_per_unit, (int, float)):
                self._unit_per_px = 1 / float(px_per_unit)
                self._px_per_unit = float(px_per_unit)
            else:
                if px_per_unit is None:
                    raise ValueError("unit_per_px and px_per_unit can't both be None at the same"
                                     " time")
                raise TypeError("px_per_unit must be an integer or a floating-point number")
        else:
            raise TypeError("unit_per_px must be an integer or a floating-point number")

        self.unit = unit

    @property
    def unit_per_px(self):
        '''Returns the calibration used in the video in units per pixel.
        
        Returns
        -------
        out : float
            Calibration in units per pixel.
        '''
        return self._unit_per_px

    @unit_per_px.setter
    def unit_per_px(self, value):
        '''Defines the calibration in units per pixel.
        
        Parameters
        ----------
        value : int, float
            New calibration.
        '''
        if not isinstance(value, (int, float)):
            raise TypeError("value must be an integer or a floating point number")
        self._unit_per_px = float(value)
        self._px_per_unit = 1 / float(value)

    @property
    def px_per_unit(self):
        '''Returns the calibration used in the video in pixels per unit.
        
        Returns
        -------
        out : float
            Calibration in pixels per unit.
        '''
        return self._px_per_unit

    @px_per_unit.setter
    def px_per_unit(self, value):
        '''Defines of the calibration in pixels per unit.
        
        Parameters
        ----------
        value : int, float
            New calibration.
        '''
        if not isinstance(value, (int, float)):
            raise TypeError("value must be an integer or a floating point number")
        self._unit_per_px = 1 / float(value)
        self._px_per_unit = float(value)



class FittedEllipses:
    '''Elements related to the ellipse fitted on the selection of the TrackedParticle using a
     Tracking method.
    '''
    def __init__(self, major_length=None, minor_length=None):
        '''Initializes a FittedEllipses instance.
        
        Parameters
        ----------
        major_length : int, float or array of int or float
            Length of the largest diameter for the each of the ellipses.
        minor_length : int, float or array of int or float
            Length of the smallest diameter for the each of the ellipses.
        calibration : Calibration
            Calibration from the initialized unit to a calibrated one.  
        '''

        self._major_length = validate_array(major_length, "major_length")
        self._minor_length = validate_array(minor_length, "minor_length")

        if len(self.major_length) != len(self.minor_length):
            raise ValueError("major_length and minor_length must have the same number of elements")

        self._u_typ_maj_len = ["fixed", 2]
        self._u_typ_min_len = ["fixed", 2]

    @property
    def major_length(self):
        '''Returns the major length of each of the ellipses in initialized unit (e.g. pixels).
        
        Returns
        -------
        out : numpy.ndarray
            Array of the major length.'''
        return self._major_length

    @property
    def minor_length(self):
        '''Returns the minor length of each of the ellipses in initialized unit (e.g. pixels).
        
        Returns
        -------
        out : numpy.ndarray
            Array of the minor length.'''
        return self._minor_length

    @property
    def size(self):
        '''Returns the total number of ellipses consider in this FittedEllipses object.
        
        Returns
        -------
        out : int
            Number of ellipses and size of the array major and minor lengths.
        '''
        return len(self.major_length)

    @major_length.setter
    def major_length(self, new_major):
        '''Defines the new major length for each ellipses.
         
        Parameters
        ----------
        new_major : int, float or array of int or float
            New array of the major lengths.
        '''
        new_major = validate_array(new_major, "new_major")
        if len(new_major)!=self.size:
            raise ValueError("new_major must have the same length as the previous one")
        self._major_length = new_major

    @minor_length.setter
    def minor_length(self, new_minor):
        '''Defines the new minor length for each ellipses.
         
        Parameters
        ----------
        new_minor : int, float or array of int or float
            New array of the minor lengths.
        '''
        new_minor = validate_array(new_minor, "new_minor")
        if len(new_minor)!=self.size:
            raise ValueError("new_minor must have the same length as the previous one")
        self._minor_length = new_minor

    @property
    def u_type_major_length(self):
        '''Returns the type and value of the uncertainty for the major_length in pixels.
        
        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value.
        '''
        return self._u_typ_maj_len

    def set_u_type_major_length(self, value, fixed=True):
        '''Defines the type and value of the uncertainty associated to the major_length in pixels.

        Parameters
        ----------
        value : int, float
            Value of the uncertainty to apply.
        fixed : bool
            Defines if the uncertainty is fixed or a percentage of the measurement considered
        '''
        if fixed:
            self._u_typ_maj_len = ["fixed", value]
        else:
            self._u_typ_maj_len = ["percentage", abs(value)]

    @property
    def u_major_length(self):
        '''Returns the uncertainty for the major_length in pixels.
        
        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value.
        '''
        if self._u_typ_maj_len[0] == "fixed":
            return np.ones_like(self.major_length) * self._u_typ_maj_len[1]
        return self.major_length * (self._u_typ_maj_len[1]/100)

    @property
    def u_type_minor_length(self):
        '''Returns the type and value of the uncertainty for the minor_length in pixels.
        
        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value.
        '''
        return self._u_typ_min_len

    def set_u_type_minor_length(self, value, fixed=True):
        '''Defines the type and value of the uncertainty associated to the minor_length in pixels.

        Parameters
        ----------
        value : int, float
            Value of the uncertainty to apply.
        fixed : bool
            Defines if the uncertainty is fixed or a percentage of the measurement considered
        '''
        if fixed:
            self._u_typ_min_len = ["fixed", value]
        else:
            self._u_typ_min_len = ["percentage", abs(value)]

    @property
    def u_minor_length(self):
        '''Returns the uncertainty for the minor_length in pixels.
        
        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value.
        '''
        if self._u_typ_min_len[0] == "fixed":
            return np.ones_like(self.minor_length) * self._u_typ_min_len[1]
        return self.minor_length * (self._u_typ_min_len[1]/100)

    @property
    def aspect_ratio(self):
        '''Returns the aspect-ratio for each pair of major and minor length.
        
        Returns
        -------
        out : numpy.ndarray
            Array of the aspect-ratio.
        '''
        return self.major_length/self.minor_length

    @property
    def u_aspect_ratio(self):
        '''Returns the uncertainty for the aspect-ratio of the ellipses.
        
        Returns
        -------
        out : list
            List of the type of uncertainty applied and its associated value.
        '''
        temp = ((self.u_major_length/self.major_length)**2 +
                (self.u_minor_length/self.minor_length)**2)**0.5
        return self.aspect_ratio * temp



# def __item_load(file, key, value):
#     '''[To Complete]
#     '''
#     pass



# def load_vid(file_path):
#     '''
#     Load a VideoStudied instance from the selected file

#     Parameters:
#     -----------
#     file_path : str
#         Path to the VideoStudied instance to load
#     '''
#     with h5py.File(file_path, 'r') as file:
#         outer_instance = VideoStudied("", 0)

#         for key, value in file.items():
#             __item_load(file, key, value)

#             if isinstance(getattr(outer_instance, key), TrackedParticle):
#                 inner_instance = TrackedParticle("", None, "")
#                 for inner_key, inner_value in value.items():
#                     setattr(inner_instance, inner_key, inner_value[()])
#                 setattr(outer_instance, key, inner_instance)
#             else:
#                 setattr(outer_instance, key, value[()])

#     return outer_instance

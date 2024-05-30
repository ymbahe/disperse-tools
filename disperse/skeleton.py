"""Skeleton class to represent a DisPerSE skeleton instance."""

from pdb import set_trace
import re
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from disperse.timestamp import TimeStamp
from pyread_eagle import EagleSnapshot

import pickle
import traceback

cosmo = FlatLambdaCDM(H0=72, Om0=0.27, Tcmb0=2.725)  # DisPerSE standard
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class Skeleton:
    """Representation of a DisPerSE skeleton.

    Parameters
    ----------
    filename : str
        The name of the .a.NDskl file from DisPerSE that contains the
        skeleton data.
    """
    def __init__(self, filename, verbose=False, sampling_factor=1.0,
                 shift_coordinates_to_cell_centres=True,
                 periodic_wrapping=False, raw_file=True):

        if not raw_file:
            self.load_from_pickle(filename)
            return

        with open(filename, 'r') as f:

            # The first line is just the header
            line = f.readline()
            if line != 'ANDSKEL\n':
                raise ValueError("Invalid input file.")
            self.verbose = verbose
            self.sampling_factor = sampling_factor

            self.filename = filename

            # Read number of dimensions
            self.n_dim = int(f.readline())
            if verbose:
                print(f"Instantiating skeleton, {self.n_dim} dimensions...")

            # Empty "no comment" line...
            f.readline()

            # Bounding box:
            line = f.readline()
            rr = numbers_from_string(line)
            if len(rr) != 2 * self.n_dim:
                raise ValueError("Invalid size of bounding box.")
            self.bbox = np.zeros((self.n_dim, 2))
            for idim in range(self.n_dim):
                self.bbox[idim, 0] = float(rr[idim])
                self.bbox[idim, 1] = float(rr[idim + self.n_dim])
            self.bbox *= sampling_factor
            
            # Critical points...
            line = f.readline()
            if line != '[CRITICAL POINTS]\n':
                raise ValueError(
                    f"Expected to read '[CRITICAL POINTS]', got '{line}'!")
            self.n_cp = int(f.readline())
            if verbose:
                print(f"Expecting {self.n_cp} critical points...")

            # Now that we know how many Critical Points to expect, we can 
            # initialize the data structure that holds all their information
            self._initialize_cp_structure()

            # We also need to initialize the list that holds all the 
            # critical point --> filament links:
            self.cp_filaments = FlexArray((self.n_cp * 4, 2), dtype=int)

            # Read and process data block for each critical point in turn...
            for icp in range(self.n_cp):
                if icp % 1000 == 0 and self.verbose:
                    print(f"Loading CP {icp}...")
                cp_dict = self._read_cp_data(f, icp)

                # Now we need to unpack these values...
                for field in [
                    'Type', 'Coordinates', 'Value', 'PairID', 'BoundaryFlag',
                    'NumberOfFilaments'
                ]:
                    self.cp_data[field][icp, ...] = cp_dict[field] 

                # First, record where in the full list this critical point's
                # filaments will be stored...
                curr_index = self.cp_filaments.next_index
                cp_nfil = cp_dict['NumberOfFilaments']
                self.cp_data['CPFilamentsOffset'][icp] = curr_index
                self.cp_data['CPFilamentsEnd'][icp] = curr_index + cp_nfil

                # ... and now put the data there
                for ifil in range(cp_dict['NumberOfFilaments']):
                    self.cp_filaments.append(cp_dict['Filaments'][ifil])

            self.cp_data['Coordinates'] *= self.sampling_factor
            if shift_coordinates_to_cell_centres:
                self.cp_data['Coordinates'] += self.sampling_factor / 2
            if periodic_wrapping:
                boxsize = self.bbox[:, 1] - self.bbox[:, 0]
                self.cp_data['Coordinates'] %= boxsize

            # Shrink cp_filaments to the number of actual elements
            self.cp_filaments.shrink()
            self.n_cp_filaments = self.cp_filaments.allocated_size

            # Yay! Done with critical points (for now...). Next up: filaments
            line = f.readline()
            if line != '[FILAMENTS]\n':
                raise ValueError(
                    f"Expected to read '[FILAMENTS]', got '{line}'! ")
            
            # Read the number of filaments. This must be exactly half the
            # number of critical point - filament connections read above,
            # because each filament connects exactly two CPs. If not -- bad.
            self.n_filaments = int(f.readline())
            if self.n_filaments != self.n_cp_filaments / 2:
                raise ValueError(
                    f"Read {self.n_cp_filaments} CP-filament "
                    f"connections, but there are {self.n_filaments} "
                    f"filaments. This does not add up!"
                )

            # Initialize the data structures for filaments and their sampling
            # points
            self._initialize_filament_structure()
            self._initialize_filament_sample_structure()

            # Read filaments one-by-one
            for ifil in range(self.n_filaments):
                if ifil % 1000 == 0 and self.verbose:
                    print(f"Loading Filament {ifil}...")
                fil_dict = self._read_filament_data(f, ifil)

                # Now we have to unpack these values...
                for field in ['CriticalPoints', 'NumberOfSamplingPoints']:
                    self.filament_data[field][ifil, ...] = fil_dict[field]

                # First record where in the full sampling point list this
                # filament's data will be stored...
                curr_index = self.sampling_data['Coordinates'].next_index
                end_index = curr_index + fil_dict['NumberOfSamplingPoints']
                self.filament_data['SamplingPointsOffset'][ifil] = curr_index
                self.filament_data['SamplingPointsEnd'][ifil] = end_index

                # ... and now get the data there
                for isample in range(fil_dict['NumberOfSamplingPoints']):
                    sample_coords = fil_dict['SamplingPoints'][isample]
                    self.sampling_data['Coordinates'].append(sample_coords)
                
            # We can now truncate the sampling coordinates to actual size
            self.sampling_data['Coordinates'].shrink()
            self.sampling_data['Coordinates'].data *= self.sampling_factor
            if shift_coordinates_to_cell_centres:
                self.sampling_data['Coordinates'].data += (
                    self.sampling_factor/2)
            self.n_sample = self.sampling_data['Coordinates'].data.shape[0]

            # Periodic wrapping of sampling data if needed
            if periodic_wrapping:
                boxsize = self.bbox[0, 1] - self.bbox[0, 0]
                self.sampling_data['Coordinates'].data = (
                    self.sampling_data['Coordinates'].data % boxsize 
                )

            # ----- Done reading basic data. Now comes ancillary info... ----

            if verbose:
                print("Loading extra info for critical points...")

            line = f.readline()
            if line != '[CRITICAL POINTS DATA]\n':
                raise ValueError(
                    f"Expected to read '[CRITICAL POINTS DATA]', got "
                    f"'{line}'!"
                )
            self.n_cp_extra_fields = int(f.readline())
            n_fields = self.n_cp_extra_fields
            if verbose:
                print(f"Processing {n_fields} extra fields.")
            field_names = [None] * n_fields
            field_names_options = [
                'persistence_ratio', 'persistence_nsigmas', 'persistence',
                'robustness_ratio', 'robustness',
                'persistence_pair', 'parent_index', 'parent_log_index',
                'log_field_value', 'field_value', 'cell'
            ]
            field_names_internal = np.array([
                'PersistenceRatio', 'PersistenceNSigmas', 'Persistence',
                'RobustnessRatio', 'Robustness',
                'PersistencePair', 'ParentIndex', 'ParentLogIndex',
                'LogFieldValue', 'FieldValue', 'Cell'
            ])

            for ii in range(n_fields):
                field_name = f.readline()[:-1]
                internal_index = field_names_options.index(field_name)
                if verbose:
                    print(f"Internal index: {internal_index}")
                    print(f"Internal name: "
                          f"{field_names_internal[internal_index]}")
                field_names[ii] = field_names_internal[internal_index]
                
            # Read extra CP info, line by line...
            for icp in range(self.n_cp):
                numbers = numbers_from_string(f.readline())
                for ifield, field in enumerate(field_names):
                    self.cp_data[field][icp] = numbers[ifield]

            # And finally, filament sampling points ancillary info
            if verbose:
                print("Loading extra info for filament sampling points...")
            line = f.readline()
            if line != '[FILAMENTS DATA]\n':
                raise ValueError(
                    f"Expected to read '[FILAMENTS DATA]', not '{line}'!")

            self.n_sample_extra_fields = int(f.readline())
            n_xfields = self.n_sample_extra_fields

            xfield_names = [None] * n_xfields
            xfield_names_options = [
                'field_value', 'orientation', 'cell', 'log_field_value',
                'type', 'robustness', 'robustness_ratio'
            ]
            xfield_names_internal = np.array([
                'FieldValue', 'Orientation', 'Cell', 'LogFieldValue', 'Type',
                'Robustness', 'RobustnessRatio'
            ])

            for ii in range(n_xfields):
                xfield_name = f.readline()[:-1]
                internal_index = xfield_names_options.index(field_name)
                if verbose:
                    print(f"Internal index: {internal_index}")
                    print(f"Internal name: "
                          f"{xfield_names_internal[internal_index]}")
                xfield_names[ii] = xfield_names_internal[internal_index]

            # Read extra sampling point info, line by line...
            self._add_filament_sample_fields()
            for isample in range(self.n_sample):
                numbers = numbers_from_string(f.readline())
                for ifield, field in enumerate(xfield_names):
                    self.sampling_data[field][isample] = numbers[ifield]

            # Add info field for sample --> filament
            self.sampling_data['Filament'] = np.zeros(
                self.n_sample, dtype=int) - 1
            for ifil in range(self.n_filaments):
                off = self.filament_data['SamplingPointsOffset'][ifil]
                end = self.filament_data['SamplingPointsEnd'][ifil]
                self.sampling_data['Filament'][off:end] = ifil

    def __repr__(self):
        info = (
            f"Skeleton network with {self.n_cp} critical points and "
            f"{self.n_filaments} filaments."
        )
        return info

    def save(self, file_name):
        """Save the class as a pickle file"""
        file = open(file_name, 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    def load_from_pickle(self, file_name):
        """Load a previously constructed class from a pickle file."""
        file = open(file_name, 'rb')
        self.__dict__ = pickle.load(file)
        file.close()

    def sampling_to_xyz(self, is_cartesian=False):
        """Calculate the sampling point coordinates in Cartesian frame.

        This assumes (without check!) that the Skeleton was output in
        RA/DEC(/redshift) coordinates. If is_cartesian, then the input
        is assumed to be Cartesian already and the coordinates are just copied.

        The output is stored as 'CartesianCoordinates' keyword of
        `self.sampling_data`.

        """
        # If the input is already Cartesian, things are short and sweet:
        if is_cartesian:
            self.sampling_data['CartesianCoordinates'] = (
                self.sampling_data['Coordinates'].data)
            return

        # Compute the radial distance of all samples. In 2D, we assume
        # a value of unity.
        if self.n_dim == 2:
            samples_dist = np.ones(self.n_sample)
        else:
            samples_zred = self.sampling_data['Coordinates'].data[:, 2]
            samples_dist = cosmo.comoving_distance(samples_zred).value

        # RA and DEC are always present as first two indices in input data
        samples_ra = self.sampling_data['Coordinates'].data[:, 0]       
        samples_dec = self.sampling_data['Coordinates'].data[:, 1]

        # Transform the coordinates to Cartesian using the SkyCoord object
        sample_coords = SkyCoord(
            ra=samples_ra * u.deg, dec=samples_dec * u.deg,
            distance=samples_dist, frame='icrs'
        )
        samples_x = np.array(c.cartesian.x)
        samples_y = np.array(c.cartesian.y)
        samples_z = np.array(c.cartesian.z)
        self.sampling_data['CartesianCoordinates'] = np.vstack(
            (samples_x, samples_y, samples_z)).T

    def sampling_to_rdz(self, is_rdz=False):
        # If the input is already RaDecZ, things are short and sweet:
        if is_rdz:
            self.sampling_data['RaDecZCoordinates'] = (
                self.sampling_data['Coordinates'].data)
            return

        # Make sure that there is a z coordinate -- if not, set to 1
        samples_x = self.sampling_data['Coordinates'].data[:, 0]
        samples_y = self.sampling_data['Coordinates'].data[:, 1]

        if self.n_dim == 2:
            samples_z = np.sqrt(1.0 - samples_x**2 - samples_y**2)
        else:
            samples_z = self.sampling_data['Coordinates'].data[:, 2]

        # Transform to RaDecDist via SkyCoord
        c = SkyCoord(
            x=samples_x, y=samples_y, z=samples_z,
            representation_type='cartesian', unit='Mpc'
        )
        ra = np.array(c.spherical.lon)
        dec = np.array(c.spherical.lat)
        distance = np.array(c.spherical.distance)

        # Convert distance --> redshift if 3D
        if self.n_dim == 3:
            zred = z_at_value(cosmo.comoving_distance, distance*u.Mpc)
            self.sampling_data['RaDecZCoordinates'] = np.vstack(
                (ra, dec, zred)).T
        else:
            self.sampling_data['RaDecZCoordinates'] = np.vstack(
                (ra, dec)).T
        
    def mask_segments(self, centres, radii, periodic=False, use_simple=False):
        """Mask filament segments that lie within exclusion zones.

        This is intended to exclude from analysis the part of filaments that
        lie within the virial radius of groups or clusters. It can however
        be used flexibly.

        Parameters
        ----------
        centres : array (N, Ndim)
            The centres of the exclusion spheres. Ndim must be consistent with
            the number of dimensions of the skeleton.
        radii : array (N)
            For each exclusion sphere, the radius within which segments are
            masked.
        periodic : bool, optional
            Apply periodic wrapping in finding the region around each exclusion
            centre (default: False, no wrapping applied).

        Returns
        -------
        None.

        The result is stored in two boolean arrays within `sampling_data`:
            - 'Mask': for each sampling point, whether it is masked
            - 'SegmentMask': for each segment, whether it is masked (i.e.
               whether either of the sampling points at its ends are masked).

        """
        # Build a tree from exclusion centres (with periodicity if desired)
        boxsize = self.bbox[0, 1] - self.bbox[0, 0]
        if use_simple:
            coords = self.sampling_data['SimpleCoordinates']
        else:
            coords = self.sampling_data['CartesianCoordinates']

        if periodic:
            sample_tree = cKDTree(coords, boxsize=boxsize)
        else:
            sample_tree = cKDTree(coords)

        # Initialize sampling point mask -- all by default unmasked (False)
        n_samples = coords.shape[0]
        sp_mask = np.zeros(n_samples, dtype=bool)

        # For each exclusion sphere, find and mark masked sampling points
        n_spheres = centres.shape[0]
        for isphere in range(n_spheres):
            cen = centres[isphere, :]
            ngbs = sample_tree.query_ball_point(cen, radii[isphere])
            sp_mask[ngbs] = True

        # Find masked *segments* -- all those where either start or end point
        # is masked. In practice, this just means masking all segments with
        # index 1 below masked segments, as those with a masked start point
        # are automatically marked as masked already. Two corner cases could
        # be relevant, but are not in practice:
        # (1) the 'fake' segment between two filaments -- may be marked as
        #     masked, but does not matter
        # (2) the very last segment, which may be marked as masked if the first
        #     sampling point is masked. Again, this is a fake segment so it
        #     does not matter.
        segment_mask = np.copy(sp_mask)
        ind_masked = np.nonzero(sp_mask == True)[0]
        segment_mask[ind_masked - 1] = True
 
        if use_simple:
            self.sampling_data['SegmentMaskSimple'] = segment_mask
            self.sampling_data['MaskSimple'] = sp_mask
        else:
            self.sampling_data['SegmentMask'] = segment_mask
            self.sampling_data['Mask'] = sp_mask

    def plot_filaments(self, ax, zrange=None, xyz=False, rdz=False,
                       cut_index=2, plot_indices=[0, 1, 2],
                       suppress_wrapping=False, plot_3d=False,
                       critical_points=None, robustness_ratio_threshold=None,
                       filament_indices=None, use_simple=False,
                       include_masked=False,
                       label_filaments=False, **kwargs):
        """Plot the filament network to a given axis.

        Parameters
        ----------
        ax : Axes object
            The axis to which the filaments will be plotted.
        zrange : ndarray(2) or None, optional
            If not None (default), the min and max value of z to plot.
            Only filaments that have at least one critical point in the
            selected range will be plotted. This has no effect for a 2D
            skeleton.
        xyz : bool, optional
            Plot the filaments in Cartesian coordinates, even if the skeleton
            itself uses RaDecZ (default: False). This requires that the
            sample coordinates have been converted to Cartesian beforehand.
        rdz : bool, optional
            Plot the filaments in RaDecZ coordinates, even if the skeleton
            itself uses Cartesian (default: False). This requires that the
            sample coordinates have been converted to RaDecZ beforehand.
        cut_index : int, optional
            The coordinate index to use for trimming the filament sample
            if `zrange` is specified. By default, this is 2.
        plot_indices : array(2), optional
            The coordinate indices to plot on the x- and y-axes, respectively.
            By default, this is [0, 1], i.e. plot x-y or RA-DEC.
        """
        # Load the appropriate sampling point coordinates
        if xyz:
            coords = self.sampling_data['CartesianCoordinates']
        elif rdz:
            coords = self.sampling_data['RaDecZCoordinates']
        elif use_simple:
            coords = self.sampling_data['SimpleCoordinates']
            if not include_masked:
                mask = self.sampling_data['MaskSimple']
        else:
            coords = self.sampling_data['Coordinates'].data
            if not include_masked:
                mask = self.sampling_data['Mask']

        if use_simple:
            n_filaments = self.n_filaments_simple
            sampling_offset = self.filament_data['SamplingPointsOffsetSimple']
            sampling_end = self.filament_data['SamplingPointsEndSimple']
        else:
            n_filaments = self.n_filaments
            sampling_offset = self.filament_data['SamplingPointsOffset']
            sampling_end = self.filament_data['SamplingPointsEnd']

        # Check whether we use a predefined color, or cycle per filament...
        if 'color' in kwargs:
            color = kwargs['color']
            del kwargs['color']
            reset_color = False
        else:
            reset_color = True

        # Loop through individual filaments. We keep a counter of plotted
        # filaments in case this turns out to be useful.
        current_plotted_filament = -1
        xlength = self.bbox[plot_indices[0]][1] - self.bbox[plot_indices[0]][0]
        ylength = self.bbox[plot_indices[1]][1] - self.bbox[plot_indices[1]][0]
        if self.n_dim == 3:
            zlength = (
                self.bbox[plot_indices[2]][1]
                - self.bbox[plot_indices[2]][0]
            )

        for ifil in range(n_filaments):
            if reset_color:
                color = colors[ifil % len(colors)]
            cp_indices = self.filament_data['CriticalPoints'][ifil, :]
            if critical_points is not None:
                if (cp_indices[0] not in critical_points and
                    cp_indices[1] not in critical_points):
                    continue
            if filament_indices is not None:
                if ifil not in filament_indices:
                    continue

            offset = sampling_offset[ifil]
            end = sampling_end[ifil]

            if robustness_ratio_threshold is not None:
                if (
                    np.mean(self.sampling_data['RobustnessRatio'][offset:end])
                    < robustness_ratio_threshold):
                    continue

            # Skip if current filament is completely outside target z range
            if self.n_dim == 3 and zrange is not None:
                z_samples = coords[offset:end, cut_index]
                if (np.min(z_samples) > zrange[1] or
                    np.max(z_samples) < zrange[0]):
                    continue

            # Check whether segments of this filament are masked
            if not include_masked:
                fil_mask = mask[offset:end]
                ind_unmasked = offset + np.nonzero(fil_mask == False)[0]
                #if np.count_nonzero(fil_mask == True) > 0: set_trace()
            else:
                ind_unmasked = np.arange(offset, end)
        
            # If we get here, we will plot the filament, so increase counter
            if len(ind_unmasked) > 0:
                current_plotted_filament += 1

            # Check whether this filament crosses a periodic boundary...
            xfil = coords[ind_unmasked, plot_indices[0]]
            yfil = coords[ind_unmasked, plot_indices[1]]

            if self.n_dim == 3:
                zfil = coords[ind_unmasked, cut_index]

            if len(xfil) == 0: continue

            if suppress_wrapping:
                if zrange is not None:
                    ind_cross = np.nonzero(
                        (np.abs(xfil[:-1] - xfil[1:]) > xlength*0.5) |
                        (np.abs(yfil[:-1] - yfil[1:]) > ylength*0.5) |
                        (zfil[:-1] < zrange[0]) | (zfil[:-1] > zrange[1]) |
                        (zfil[1:] < zrange[0]) | (zfil[1:] > zrange[1])
                    )[0]
                elif plot_3d:
                        ind_cross = np.nonzero(
                        (np.abs(xfil[:-1] - xfil[1:]) > xlength*0.5) |
                        (np.abs(yfil[:-1] - yfil[1:]) > ylength*0.5) |
                        (np.abs(zfil[:-1] - zfil[1:]) > zlength*0.5)
                    )[0]

                else:
                    ind_cross = np.nonzero(
                        (np.abs(xfil[:-1] - xfil[1:]) > xlength*0.5) |
                        (np.abs(yfil[:-1] - yfil[1:]) > ylength*0.5)
                    )[0]
            else:
                ind_cross = np.zeros(0)
                    
            start_points = [0]
            end_points = [len(xfil)]
            if len(ind_cross) > 0:
                start_points = np.concatenate((start_points, ind_cross+1))
                end_points = np.concatenate((end_points, ind_cross+1))
            start_points = np.sort(start_points)
            end_points = np.sort(end_points)
            n_seg = len(start_points)
            #if n_seg > 1:
            #    print(f"Filament {ifil}: {n_seg} sectors.")

            # Now loop over the 'segments' of each filament to plot (i.e., the
            # different pieces separated by periodic boundary crossings).
            for iseg in range(n_seg):
                #print(iseg, n_seg)
                if end_points[iseg] - start_points[iseg] > 1:
                    if self.n_dim == 3 and plot_3d:
                        l, = ax.plot(
                            xfil[start_points[iseg] : end_points[iseg]],
                            yfil[start_points[iseg] : end_points[iseg]],
                            zfil[start_points[iseg] : end_points[iseg]],
                            color=color,
                            **kwargs
                        )
                    else:
                        l, = ax.plot(
                            xfil[start_points[iseg] : end_points[iseg]],
                            yfil[start_points[iseg] : end_points[iseg]],
                            color=color,
                            **kwargs
                        )
                if iseg == 0 and label_filaments:
                    index = start_points[iseg]
                    ax.annotate(
                        f'{ifil}',
                        xy=(xfil[index], yfil[index]),
                        xytext=(0, 0),
                        textcoords='offset points', ha='center', va='center',
                        fontsize=4, color=color,
                        bbox={'edgecolor':color, 'facecolor':'white',
                              'boxstyle': 'circle', 'alpha': 0.8,
                              'pad': 0.25}
                    )
                if iseg == n_seg - 1 and label_filaments:
                    index = end_points[iseg] - 1
                    #print(f"End {iseg}")
                    ax.annotate(
                        f'{ifil}',
                        xy=(xfil[index], yfil[index]),
                        xytext=(0, 0),
                        textcoords='offset points', ha='center', va='center',
                        fontsize=4, color=color,
                        bbox={'edgecolor':color, 'facecolor':'black',
                              'boxstyle': 'circle', 'alpha': 0.8,
                              'pad': 0.25}
                    )

    def exp_find_filament_neighbours(
            self, snapshot_file_name, distance, ind_filaments=None,
            use_simple=False
        ):
        ts = TimeStamp()
        if np.isscalar(ind_filaments):
            ind_filaments = np.array([ind_filaments])
        boxsize = self.bbox[0, 1] - self.bbox[0, 0]
            
        # Create aliases for the appropriate filament data, so that we can
        # process both the raw and simplified network in the same way. After
        # this point, there is no more distinction between the two cases.
        if use_simple:
            n_filaments = self.n_filaments_simple
            n_samples = self.n_sample_simple
            sample_coords = self.sampling_data['SimpleCoordinates']
            offsets = self.filament_data['SamplingPointsOffsetSimple']
            ends = self.filament_data['SamplingPointsEndSimple']
            sample_filament = self.sampling_data['FilamentSimple']
        else:
            n_filaments = self.n_filaments
            n_samples = self.n_sample
            sample_coords = self.sampling_data['CartesianCoordinates']
            offsets = self.filament_data['SamplingPointsOffset']
            ends = self.filament_data['SamplingPointsEnd']
            sample_filament = self.sampling_data['Filament']
        ts.set_time("Load filament data")
                    
        # To enable searching around sampling points, we need to find those
        # belonging to selected filaments (if only processing a subset).
        # For consistency, we also create a full list of sample points to be
        # used if analysing the full filament network.
        if ind_filaments is not None:
            mask_sample = np.zeros(n_samples, dtype=bool)
            for ifil in ind_filaments:
                mask_sample[offsets[ifil] : ends[ifil]] = True
            ind_sample = np.nonzero(mask_sample)[0]
        else:
            ind_sample = np.arange(n_samples)
        n_sample = len(ind_sample)
        ts.set_time("Make sample point list")

        if with_sample_spheres:
            sphere_ngb, sphere_cyl = self.find_sphere_neighbours(
                sample_coords[ind_sample, :], distance)
        
        self.find_sector_neighbours(snapshot_file_name, distance)

        
    def find_sphere_neighbours(
        self, snapshot_file_name, centres, distance):
        "Dummy for finding neighbours around spheres"
        return None, None
        
    def find_sector_neighbours(
            self, snapshot_file_name, distance, p1, p2, ptype=1):
        # Load particles
        pvec = p2 - p1
        pmid = (p1 + p2) / 2
        dp = np.linalg.norm(pvec)
        radius = np.sqrt(distance**2 + (dp/2)**2)
        
        snap = EagleSnapshot(snapshot_file_name)
        snap.select_region(
            dp[0]-radius, dp[0]+radius,
            dp[1]-radius, dp[1]+radius,
            dp[2]-radius, dp[2]+radius
        )
        coords = snap.read_dataset(ptype, "Coordinates")
        ind_cyl, d_cyl = self._find_cylinder_member_points(
            p1, p2, coords, distance, use_tree=False)
        return ind_cyl, d_cyl    
        

    def find_points_near_filaments(
        self, coords, distance, periodic=False, ind_filaments=None,
        use_simple=False, with_sample_spheres=True, verbose=False,
        tree_points=None, individual_filaments=False,
        individual_segments=False
        ):
        """Test which points of an input sample lie near a filament.

        "Near" a filament means either within a cylinder of radius `distance`
        around at least one filament segment, or within a sphere of radius
        `distance` around one sample point. The latter is to provide a buffer
        at kinks in the filament.

        This requires the sampling coordinates to have been converted to
        Cartesian with `sampling_to_xyz()`. 

        Parameters
        ----------
        coords : ndarray
            The coordinates of the test sample. They must be in the same
            units as the internal coordinates of the skeleton.
        distance : float
            The maximum distance of a "member" point from a filament.
        periodic : bool, optional
            Whether or not to take periodic boundary conditions into account.
            Default is False, so no wrapping is performed.
        ind_filaments : array-like, optional
            If specified, only the listed filaments are analysed. If None
            (default), all filaments are analysed.
        use_simple : bool, optional
            Switch to use the simplified rather than raw filament network.
            Default is False (use raw).
        with_sample_spheres : bool, optional
            Switch to add a search in a sphere of `distance` around the
            filaments' sampling points (default: True). Set to False to only
            search for points in a cylinder around each segment.
        verbose : bool, optional
            Enable more detailed log output (default: False).
        tree_points : cKDTree, optional
            Tree of the points to search against. If None, it is constructed
            internally, at potentially significant time and memory cost.
        individual_filaments : bool, optional
            Switch to record (all) neighbours for each filament separately,
            instead of only neighbours of the total filament network and their
            closest filament. Default: False.
        individual_segments : bool, optional
            Switch to record neighbours of individual filament segments. It is
            unclear whether this works without also switching on
            `individual_filaments`...

        Returns
        -------
        ind_ngb : ndarray(int)
            The indices within `coords` (along the first axis) of points that
            are within `distance` of a filament.
        min_dist : ndarray(float)
            For each input point, the minimum distance to a filament. NaN for
            points that are not near a filament.
        ind_fil : ndarray(int)
            Index of the closest filament to each point
        num_match : ndarray(int)
            Number of filaments to which each point was matched. **Currently
            not well implemented, e.g. it counts the caps and cylinders
            independently!**

        """
        ts = TimeStamp()
        n_points = coords.shape[0]
        if np.isscalar(ind_filaments):
            ind_filaments = np.array([ind_filaments])
        boxsize = self.bbox[0, 1] - self.bbox[0, 0]
            
        # Create aliases for the appropriate filament data, so that we can
        # process both the raw and simplified network in the same way. After
        # this point, there is no more distinction between the two cases.
        if use_simple:
            n_filaments = self.n_filaments_simple
            n_samples = self.n_sample_simple
            sample_coords = self.sampling_data['SimpleCoordinates']
            offsets = self.filament_data['SamplingPointsOffsetSimple']
            ends = self.filament_data['SamplingPointsEndSimple']
            sample_filament = self.sampling_data['FilamentSimple']
        else:
            n_filaments = self.n_filaments
            n_samples = self.n_sample
            sample_coords = self.sampling_data['CartesianCoordinates']
            offsets = self.filament_data['SamplingPointsOffset']
            ends = self.filament_data['SamplingPointsEnd']
            sample_filament = self.sampling_data['Filament']
        ts.set_time("Load filament data")
                    
        # To enable searching around sampling points, we need to find those
        # belonging to selected filaments (if only processing a subset).
        # For consistency, we also create a full list of sample points to be
        # used if analysing the full filament network.
        if ind_filaments is not None:
            mask_sample = np.zeros(n_samples, dtype=bool)
            for ifil in ind_filaments:
                mask_sample[offsets[ifil] : ends[ifil]] = True
            ind_sample = np.nonzero(mask_sample)[0]
        else:
            ind_sample = np.arange(n_samples)
        n_sample = len(ind_sample)
        ts.set_time("Make sample point list")
        
        # If no tree for the target points was provided, we need to build it
        # internally now. Warning, this might take a while...
        if tree_points is None:
            if verbose:
                print(
                    f"Building tree of {n_points} input points and "
                    f"{n_sample} filament sampling points...",
                    end='', flush=True
                )
            if periodic:
                tree_points = cKDTree(coords, boxsize=boxsize)
            else:
                tree_points = cKDTree(coords)
            ts.set_time("Building target point tree")
            if verbose:
                print(f"...done ({ts.get_time():.2f} sec.)", flush=True)

        # Set up arrays recording whether a point is near a filament, and 
        # what the minimum filament distance is
        flag = np.zeros(n_points, dtype=bool)
        min_dist = np.zeros(n_points) + 1000.0
        ind_fil = np.zeros(n_points, dtype=int) - 1
        num_match = np.zeros(n_points, dtype=int)

        # If we want to collect neighbours broken down by individual filaments
        # and/or segments, set up the (initially empty) lists to hold these too
        if individual_filaments:
            ngbs_per_filament = []
            dcyl_per_filament = []
        if individual_segments:
            ngbs_per_segment = []
            dcyl_per_segment = []
        
        ts.set_time('Setup output')

        # =============================================================
        # The easy part: find all points within r from a sampling point
        # =============================================================

        if with_sample_spheres:
            if verbose:
                print(f"Testing membership near sampling points...")

            # Build a tree of the sampling points -- usually quick
            if periodic:
                tree_samples = cKDTree(
                    sample_coords[ind_sample, :], boxsize=boxsize)
            else:
                tree_samples = cKDTree(sample_coords[ind_sample, :])
                
            # Find all neighbours around all sampling points. Note, this
            # builds a list-of-lists and may get inefficient for very large
            # numbers of sampling points.
            ngb_lol = tree_samples.query_ball_tree(tree_points, distance)

            # Process each sampling point separately to mark its neighbours
            for iisample, isample in enumerate(ind_sample):
                ngbs = np.array(ngb_lol[iisample])
                if len(ngbs) == 0:
                    continue

                # Mark all neighbours of the point as "near the network"
                flag[ngbs] = True

                # Update the allocation of closest filament for those
                # neighbours that are closer to the current sampling point than
                # any previous ones
                dist_ngbs = np.linalg.norm(
                    coords[ngbs, :] - sample_coords[isample, :], axis=1)
                subind_best = np.nonzero(dist_ngbs < min_dist[ngbs])
                ind_best = ngbs[subind_best]
                min_dist[ind_best] = dist_ngbs[subind_best]
                ind_fil[ind_best] = sample_filament[isample]

            ts.set_time('Neighbours around sampling spheres')
                
        # =================================================================
        # The less easy part: find all points within a cylinder around each
        # filament segment...
        # =================================================================

        if verbose:
            print(f"Testing membership in cylinder segments...")

        # Process each filament individually -- this starts a long loop
        for ifil in range(n_filaments):
            if ind_filaments is not None:
                if ifil not in ind_filaments: continue
            tss = TimeStamp()
                
            # Arrays to record membership and distance for each target point,
            # just for this filament
            if individual_filaments:
                flag_filament = np.zeros(n_points, dtype=bool)
                dcyl_filament = np.zeros(n_points) + 1000

            # If we want to record membership broken down by segment, we need
            # to set up a second layer of containers for all the segments in
            # this current filament. Note that we don't know the number of
            # actual segments (sectors) in advance, because some may be split
            # by periodic wrapping
            if individual_segments:
                ngbs_per_segment_ifil = []
                dcyl_per_segment_ifil = []
                flag_segment = np.zeros(n_points, dtype=bool)
                dcyl_segment = np.zeros(n_points) + 1000
                
            tss.set_time(f'Setup filament')

            # Loop through each segment of this filament, using the index of
            # its start sampling point as a label. That is why we subtract
            # 1 from `SamplingPointsEnd`, to not include the unphysical
            # "bridge" to the subsequent filament.

            for iseg in range(offsets[ifil], ends[ifil] - 1):
                tsss = TimeStamp()
                p1 = sample_coords[iseg, :]
                p2 = sample_coords[iseg+1, :]
                dp = p2 - p1
                tsss.set_time('Setup')
 
                if periodic:
                    # With periodic boundaries, each segment can be broken
                    # into multiple sectors. Use master lists to collect
                    # membership and distance across those, even though there
                    # should not be any overlap in practice...
                    if individual_segments:
                        flag_segment[:] = False
                        dcyl_segment[:] = 1000
                    tsss.set_time("Initialize segment list")
                        
                    # Alright. Need to check for each dimension whether the
                    # segment crosses one or more periodic boundaries.
                    # Depending on how often this happens, we end up with 1-8
                    # sectors created from the current point pair.
                    # Initialise two lists to hold all these pairings
                    a_list = [p1]
                    b_list = [p2]
                    for idim in range(self.n_dim):
                        if np.abs(dp[idim]) > boxsize / 2:
                            a_list, b_list = self._split_points(
                                idim, boxsize, a_list=a_list, b_list=b_list)
                    tsss.set_time("Splitting")    
                    
                    # We now have the full list of pairings that need to be
                    # tested for cylinder membership.
                    tsss.add_counters(
                        ["Find cylinder members", "Record filament ngbs",
                         "Record segment ngbs"]
                    )
                    for ia, ib in zip(a_list, b_list):
                        tsss.start_time()
                        ind_cyl, d_cyl = self._find_cylinder_members(
                            ia, ib, coords, distance, ifil=ifil,
                            tree_points=tree_points,
                            flag=flag, min_dist=min_dist, ind_fil=ind_fil)
                        num_match[ind_cyl] += 1
                        tsss.increase_time("Find cylinder members")
                        
                        if individual_filaments:
                            flag_filament[ind_cyl] = True

                            # Update the closest cylindrical radius within
                            # the current filament
                            subind_best = np.nonzero(
                                d_cyl < dcyl_filament[ind_cyl])[0]
                            dcyl_filament[ind_cyl[subind_best]] = (
                                d_cyl[subind_best])
                        tsss.increase_time("Record filament ngbs")

                        if individual_segments:
                            flag_segment[ind_cyl] = True

                            # Update closest cylindrical radius within
                            # current segment (non-trivial in case of
                            # splitting due to periodic wrapping)
                            subind_best = np.nonzero(
                                d_cyl < dcyl_segment[ind_cyl])[0]
                            dcyl_segment[ind_cyl[subind_best]] = (
                                d_cyl[subind_best])
                        tsss.increase_time("Record segment ngbs")

                else:
                    # Without periodic wrapping, things are easy.
                    ngbs_segment, dcyl_segment = self._find_cylinder_members(
                        p1, p2, coords, distance, ifil=ifil,
                        tree_points=tree_points,
                        flag=flag, min_dist=min_dist, ind_fil=ind_fil
                    )
                    tsss.set_time("Find cylinder members")
                    num_match[ngbs_segment] += 1

                    if individual_filaments:
                        flag_filament[ngbs_segment] = True

                        # Update closest cylindrical radius within the
                        # current filament
                        subind_best = np.nonzero(
                            dcyl_segment < dcyl_filament[ngbs_segment])[0]
                        dcyl_filament[ngbs_segment[subind_best]] = (
                            dcyl_segment[subind_best])
                        tsss.set_time("Record filament ngbs")

                        # We do not need to update the distance for individual
                        # segments, because without periodic wrapping there
                        # is only one sector per segment.
                        
                # Ends periodic/non-periodic distinction; we have all
                # neighbours and their distances for the current segment.
                # Gather them if we are interested in them.
                if individual_segments:

                    # Need to still extract actual neighbours for periodic
                    if periodic:
                        ngbs_segment = np.nonzero(flag_segment)[0]

                    ngbs_per_segment_ifil.append(ngbs_segment)
                    dcyl_per_segment_ifil.append(dcyl_segment[ngbs_segment])
                    tsss.set_time("Append segment data")
                    
                #tsss.print_time_usage(f'Segment {iseg}')
                tss.import_times(tsss)
                
            # Back at per-filament level -- gather info if required

            tss.set_time('Segment loop')
            if individual_filaments:
                ngbs_filament = np.nonzero(flag_filament)[0]
                dcyl_filament = dcyl_filament[ngbs_filament]
                ngbs_per_filament.append(ngbs_filament)
                dcyl_per_filament.append(dcyl_filament)
            tss.set_time("Append filament data")
            if individual_segments:
                ngbs_per_segment.append(ngbs_per_segment_ifil)
                dcyl_per_segment.append(dcyl_per_segment_ifil)
            tss.set_time("Append filament-segment data")

            #tss.print_time_usage(f'Filament {ifil}')

        # End of loop over filaments
        ts.import_times(tss)

        ind_ngb = np.nonzero(flag)[0]

        return_list = [ind_ngb, min_dist, ind_fil, num_match]
        if individual_filaments:
            return_list.append(ngbs_per_filament)
            return_list.append(dcyl_per_filament)
        if individual_segments:
            return_list.append(ngbs_per_segment)
            return_list.append(dcyl_per_segment)

        ts.set_time("Finishing")
        #ts.print_time_usage()
            
        return return_list

    def get_filament_sampling_points(
        self, ifil, use_simple=False, exclude_masked=True):
        """Return the indices of the sampling points for one filament.

        This is a simple helper function.

        Returns
        -------
        sample_indices : ndarray(int)
            The indices of the filament sampling points, into
            `self.sampling_data`.
        """
        if use_simple:
            offset = self.filament_data['SamplingPointsOffsetSimple'][ifil]
            end = self.filament_data['SamplingPointsEndSimple'][ifil]
        else:
            offset = self.filament_data['SamplingPointsOffset'][ifil]
            end = self.filament_data['SamplingPointsEnd'][ifil]

        indices = np.arange(offset, end)

        if exclude_masked:
            if use_simple:
                mask = self.sampling_data['MaskSimple'][indices]
            else:
                mask = self.sampling_data['Mask'][indices]
            ind_good = np.nonzero(mask == False)[0]
            indices = indices[ind_good]

        return indices

    def simplify_filaments(self, threshold_angle, periodic_wrapping=False,
        exclude_zero_length=True, exclude_masked=True):
        """Simplify the filament network by joining filaments across CPs.

        Two filaments can be joined at a critical point (CP) if the angle
        between them is greater than some threshold (defining two perfectly
        aligned filaments as at 180 degrees and a perfectly bent filament as
        0 degrees).

        Under significant development and may not work as expected.
        
        """
        sampling_coords = self.sampling_data['Coordinates'].data
        boxsize = self.bbox[0, 1] - self.bbox[0, 0]

        # Initialise a list of new (simplified) filament IDs. Initially,
        # this is just identical to the old IDs.
        new_filament_ids = np.arange(self.n_filaments, dtype=int)
        filament_pairs = []

        # Go through critical points and identify joinable filaments...
        for icp in range(self.n_cp):
            n_fil = self.cp_data['NumberOfFilaments'][icp]
            if n_fil < 2: continue

            pos_cp = self.cp_data['Coordinates'][icp, :]
            entries = np.arange(self.cp_data['CPFilamentsOffset'][icp],
                                self.cp_data['CPFilamentsEnd'][icp])
            cp_filaments = self.cp_filaments.data[entries]
            filaments = np.zeros(n_fil, dtype=int) - 1
            reference_points = np.zeros((n_fil, self.n_dim)) - 1
            use_filaments = np.zeros(n_fil, dtype=bool)

            for iientry, ientry in enumerate(cp_filaments):
                pair, ifil = ientry[0], ientry[1]
                filaments[iientry] = ifil

                # Need to find the "reference point" for this filament.
                # Currently, this is its sampling point one away from the CP.
                cp_start = self.filament_data['CriticalPoints'][ifil, 0]
                cp_end = self.filament_data['CriticalPoints'][ifil, 1]
                sampling_offset = (
                    self.filament_data['SamplingPointsOffset'][ifil])
                sampling_end = (
                    self.filament_data['SamplingPointsEnd'][ifil])
                sample_pos = sampling_coords[sampling_offset : sampling_end, :]

                use_filaments[iientry] = True
                if icp == cp_start:
                    reference_points[iientry, :] = sample_pos[1, :]
                    if exclude_masked:
                        if self.sampling_data['Mask'][sampling_offset] == True:
                            use_filaments[iientry] = False
                elif icp == cp_end:
                    if exclude_masked:
                        if self.sampling_data['Mask'][sampling_end-1] == True:
                            use_filaments[iientry] = False
                    reference_points[iientry, :] = sample_pos[-2, :]
                else:
                    raise ValueError("Critical point is neither start or end?")

            # We now have all the filaments that come to this CP and their 
            # coordinates. Find at most one pair that is continuing across it,
            # based on the pair with the largest angle between them
            # (if it is above he threshold)
            max_angle = 0
            current_pair = None
            for iifil in range(n_fil):
                if exclude_masked and use_filaments[iifil] == False: continue
                if exclude_zero_length:
                    if self.filament_data['LengthsInMpc'][filaments[iifil]] == 0:
                        continue
                vec_i = reference_points[iifil, :] - pos_cp
                if periodic_wrapping:
                    ind_low = np.nonzero(vec_i < -boxsize/2)
                    vec_i[ind_low] += boxsize
                    ind_high = np.nonzero(vec_i > boxsize/2)
                    vec_i[ind_high] -= boxsize
                len_i = np.linalg.norm(vec_i)
                for jjfil in range(n_fil):
                    if exclude_masked and use_filaments[jjfil] == False: continue
                    if exclude_zero_length:
                        if self.filament_data['LengthsInMpc'][filaments[iifil]] == 0:
                            continue
                    if jjfil == iifil: continue
                    vec_j = reference_points[jjfil, :] - pos_cp
                    if periodic_wrapping:
                        ind_low = np.nonzero(vec_j < -boxsize/2)
                        vec_j[ind_low] += boxsize
                        ind_high = np.nonzero(vec_j > boxsize/2)
                        vec_j[ind_high] -= boxsize
                    len_j = np.linalg.norm(vec_j)
                    cos_angle = np.dot(vec_i, vec_j) / (len_i * len_j)
                    if -1.00001 < cos_angle < -1:
                        cos_angle = -1
                    if 1 > cos_angle > 1.00001:
                        cos_angle = 1
                    if cos_angle < -1 or cos_angle > 1:
                        print(icp, iifil, jjfil, len_i, len_j, cos_angle)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    if angle > max_angle:
                        max_angle = angle
                        pair_fils = np.array([filaments[iifil], filaments[jjfil]])
                        current_pair = (np.min(pair_fils), np.max(pair_fils))

            # Record the best join if its angle is appropriate
            if max_angle > threshold_angle:
                filament_pairs.append(current_pair)
                #if new_filament_ids[current_pair[1]] == current_pair[1]:
                #    new_filament_ids[current_pair[1]] = current_pair[0]
                #elif new_filament_ids[current_pair[0]] == current_pair[0]:
                #    new_filament_ids[current_pair[0]] = current_pair[1]
                #else:
                #    raise ValueError("Both filament pairs already reassigned!")

        # We now have a translation list of old --> new filament IDs, but this
        # is not yet self-consistent -- there may be "chain-renamings" if,
        # e.g., 0<-->1 and 1<-->2. At the moment, filament 2 is still listed
        # as relabled to 1, when in reality it should also be 0. 

        filament_pairs = np.array(filament_pairs)
        p_unique, p_count = np.unique(
            np.ravel(filament_pairs), return_counts=True)
        filament_pair_counts = np.zeros(self.n_filaments, dtype=int)
        filament_pair_counts[p_unique] = p_count
        p_singles = p_unique[np.nonzero(p_count == 1)[0]]
        flag = np.zeros(self.n_filaments, dtype=bool)

        for ising in p_singles:
            # Has this filament already been reassigned? Then no need to
            # look at it again.
            if new_filament_ids[ising] != ising:
                continue
            curr_filament = ising
            # Walk across linked filaments. Max counter of 1000 here in case
            # of unexpected infinite-loop problems.
            for iii in range(1000):
                # Mark current filament as attached to current string
                new_filament_ids[curr_filament] = ising
                flag[curr_filament] = True

                # Have we arrived at another single filament? Then the current
                # "chain" is finished. Otherwise, we keep travelling (but we
                # need to explicitly exclude the first iteration, when we are
                # still at the starting filament).
                if filament_pair_counts[curr_filament] == 1 and iii > 0:
                    break 

                # If we get here, there is another filament that this one
                # links to. Find it and then jump to it.
                ind = np.nonzero(filament_pairs == curr_filament)
                entries = ind[0]
                pairs = filament_pairs[entries, (ind[1] + 1) % 2]
                ind_pair = np.nonzero(flag[pairs] == False)[0]

                if len(ind_pair) > 1:
                    raise ValueError("A filament cannot pair with > 1!")
                if len(ind_pair) == 0:
                    set_trace()
                    raise ValueError(
                        "How can we land at a totally un-connected filament?")                    

                curr_filament = pairs[ind_pair[0]]


        # The bit below turned out not to work reliably...
        """        
        iiter = 0
        while(True):
            iiter += 1
            new_ids_b = new_filament_ids[new_filament_ids]
            diff = np.sum(np.abs(new_ids_b - new_filament_ids))
            new_filament_ids = new_ids_b

            # Can stop if there has been no change -- then we are converged
            if diff == 0: break

            # Emergency break if we are in an infinite loop...
            if iiter == 1000:
                raise ValueError(
                    f"Filament renaming did not converge after {ii} "
                    f"iterations! Please investigate."
                )
        """


        # Make the new filament IDs dense. `unique_ids` gives the old 
        # (surviving) filament IDs for each new one. 
        unique_ids, new_filament_ids = np.unique(new_filament_ids, return_inverse=True)
        self.n_filaments_simple = len(unique_ids)

        # Need to create the new list of sampling points. In principle we
        # should also update the CP->filament links, but let's leave that
        # for now...
        self.sampling_data["SimpleCoordinates"] = np.zeros(
            (self.n_sample, self.n_dim)) - 1
        self.filament_data["SamplingPointsOffsetSimple"] = np.zeros(
            self.n_filaments_simple, dtype=int) - 1
        self.filament_data["SamplingPointsEndSimple"] = np.zeros(
            self.n_filaments_simple, dtype=int) - 1
        sampling_offset = 0

        # Some data that will be needed later
        fil_cps = self.filament_data['CriticalPoints']
        old_samples = self.sampling_data['Coordinates'].data

        for ifil in range(self.n_filaments_simple):
            ind_tothis = np.nonzero(new_filament_ids == ifil)[0]
            n_tothis = len(ind_tothis)
            fil_cps_tothis = fil_cps[ind_tothis, :]

            # Find the start and end CPs for the new combined filament.
            # Those are the ones that appear only once in the list of CPs
            # across all the old filaments that make this new one.
            cps = self.filament_data['CriticalPoints'][ind_tothis, ...]
            cps_unique, counts_unique = np.unique(
                np.ravel(cps), return_counts=True)
            if np.max(counts_unique) > 2:
                print(
                    f"WARNING!!! New filament {ifil} traverses a CP "
                    f"more than twice (N_max={np.max(counts_unique)} for CP "
                    f"{cps_unique[np.argmax(counts_unique)]})!\n"
                    f"This likely indicates the formation of a 'lassoo "
                    f"filament', but you might want to verify this explicitly."
                )
            ind_ends = np.nonzero(counts_unique != 2)[0]
            cp_ends = cps_unique[ind_ends]
            if len(ind_ends) != 2:
                set_trace()
                raise ValueError("Need exactly two ends of a filament!")
            #if np.max(counts_unique > 2):
            #    raise ValueError(
            #        "A critical point cannot occur more than twice in a "
            #        "combined filament!"
            #    )

            # Sort the CPs in the right sequence
            previous_cp = -1
            cp = cp_ends[0]

            # Deal with 'lassoo' filaments... Here we want to start at the
            # loose end (the CP covered only once) to avoid the possibility of
            # going in the wrong direction at the knot. This is a dirty fix,
            # which will not work if both ends are tied.
            ind_in_unique = np.nonzero(cps_unique == cp)[0]
            if counts_unique[ind_in_unique] > 2:
                cp = cp_ends[1]
                cp_ends[1] = cp_ends[0]
                cp_ends[0] = cp
            cp_list = [cp]

            self.filament_data['SamplingPointsOffsetSimple'][ifil] = (
                sampling_offset)

            # Loop through all old filaments that make up the current new one
            for ii in range(n_tothis):
                # Find the CP that is directly connected to the current one
                xx_pair, xx_ind = np.nonzero(cps == cp)
                xx_other = (xx_ind + 1) % 2
                cps_other = cps[xx_pair, xx_other]

                # Caveat: apart from the start/end point, each CP appears 2x!
                # So we have to make sure we are not selecting the filament
                # going back to where we just came from.
                ind_next = np.nonzero(cps_other != previous_cp)[0]
                if len(ind_next) == 0: set_trace()
                ind_next = ind_next[0]
                next_cp = cps_other[ind_next]

                # Find the (old) filament connecting these CPs
                cps_start_temp = fil_cps_tothis[:, 0]
                cps_end_temp = fil_cps_tothis[:, 1]
                ii_old_tothis = np.nonzero(
                    ((cps_start_temp == cp) & (cps_end_temp == next_cp)) |
                    ((cps_end_temp == cp) & (cps_start_temp == next_cp))
                )[0]
                if len(ii_old_tothis) != 1:
                    raise ValueError("Did not find exact filament match!")
                ii_old = ind_tothis[ii_old_tothis[0]]
                old_offset = self.filament_data['SamplingPointsOffset'][ii_old]
                old_end = self.filament_data['SamplingPointsEnd'][ii_old]
                curr_samples = old_samples[old_offset : old_end]
                #set_trace()

                # We may be traversing this filament in reverse order. If so,
                # invert it to have a continuous line of sampling points.
                if fil_cps[ii_old, 1] == cp:
                    curr_samples = np.flip(curr_samples, axis=0)

                # Unless we are at the first (old) filament for this new one,
                # we don't need the first sampling point (the start CP), as
                # it's already there from the previous (old) filament.
                if ii > 0:
                    curr_samples = curr_samples[1:, ...]

                # Now we can finally add the samples to the list!
                end = sampling_offset + curr_samples.shape[0]
                self.sampling_data['SimpleCoordinates'][sampling_offset:end, ...] = (
                    curr_samples)
                sampling_offset = end

                # Update indices for next iteration
                previous_cp = cp
                cp = next_cp
                cp_list.append(cp)

            # Safety check: we should now have arrived at the other CP end!
            if cp != cp_ends[1]:
                raise ValueError(
                    f"Should have arrived at CP {cp_ends[1]}, "
                    f"but got to {cp} instead! Please investigate."
                )
            self.filament_data['SamplingPointsEndSimple'][ifil] = (
                sampling_offset)
        self.simple_filament_ids = new_filament_ids

        # Although the new filaments cover all the old ones, there are fewer
        # sampling points because joint CPs are only retained once. So we need
        # to update the total number of sampling points too.
        self.n_sample_simple = sampling_offset
        self.sampling_data['SimpleCoordinates'] = (
            self.sampling_data['SimpleCoordinates'][:sampling_offset, ...])

        # Add info field for sample --> filament
        self.sampling_data['FilamentSimple'] = np.zeros(
            self.n_sample_simple, dtype=int) - 1
        for ifil in range(self.n_filaments_simple):
            off = self.filament_data['SamplingPointsOffsetSimple'][ifil]
            end = self.filament_data['SamplingPointsEndSimple'][ifil]
            self.sampling_data['FilamentSimple'][off:end] = ifil

    def _split_points(self, idim, boxsize, a_list, b_list):
        """Split a given list of A/B points along a periodic boundary.

        This is necessary to include all mirror pairings between two points
        for filament neighbour extraction. The input lists `a_list` and
        `b_list` are returned, expanded. For A, mirror points are added in
        original order at the beginning, for B, they are added in inverse order
        at the end. This ensures that each pair A[i]/B[i] defines a valid
        cylinder segment for which neighbours can then be identified.
        """

        ts = TimeStamp()
        
        # Step 1: define the offsets for A and B...
        a_orig = a_list[-1]
        b_orig = b_list[0]
        offset_a = np.zeros(self.n_dim)
        offset_b = np.zeros(self.n_dim)
        if a_orig[idim] > boxsize/2:
            offset_a[idim] = -boxsize
            offset_b[idim] = boxsize
        else:
            offset_a[idim] = boxsize
            offset_b[idim] = -boxsize
        ts.set_time("Define offsets")
            
        # Step 2: Mirror A & B point(s)
        a_mirror = np.copy(a_list)
        b_mirror = np.copy(b_list)
        for ii in range(len(a_list)):
            a_mirror[ii] += offset_a
            b_mirror[ii] += offset_b
        ts.set_time("Mirror")
            
        # Step 3: Add mirrored points back to original list in right order
        a_list = np.concatenate((a_mirror, a_list))
        b_list = np.concatenate((b_list, np.flip(b_mirror, axis=0)))
        ts.set_time("Update A/B lists")
        #ts.print_time_usage("_split_points")
        
        return a_list, b_list

    def _find_cylinder_member_points(
        self, p1, p2, coords, distance, use_tree=False):
        """Find out which points of an input set are within a cylinder."""
        ts = TimeStamp()
        
        # Empty return value
        empty = [np.zeros(0, dtype=int), np.zeros(0)]

        # Construct cylinder mid-point and length
        pvec = p2 - p1
        pmid = (p1 + p2) / 2
        dp = np.linalg.norm(pvec)
        ts.set_time("Setup")

        if use_tree:
            tree_points = cKDTree(coords)
            ind_candidate = tree_points.query_ball_point(
                pmid, np.sqrt((dp/2)**2 + distance**2))
            ts.set_time("Tree query")
            
            # No need to continue if there are no neighbours
            if len(ind_candidate) == 0:
                return empty
            q = coords[ind_candidate, :]
            ts.set_time(f"Extract {q.shape[0]} candidate coords")
        else:
            q = coords 
            
        # Check points against the actual cylinder.
        alpha_1 = np.dot(q - p1, pvec)
        sic_1 = np.nonzero(alpha_1 >= 0)[0]
        if len(sic_1) == 0: return empty
        alpha_2 = np.dot(q[sic_1, :] - p2, pvec)
        sic_2 = np.nonzero(alpha_2 <= 0)[0]
        if len(sic_2) == 0: return empty
        d_cyl = np.linalg.norm(
            np.cross(q[sic_1[sic_2], :] - p1, pvec), axis=1) / dp 
        sic_3 = np.nonzero(d_cyl <= distance)[0]
        if len(sic_3) == 0: return empty
        subind_in_cylinder = sic_1[sic_2[sic_3]]
        #subind_in_cylinder = np.nonzero(
        #    (alpha_1 >= 0) & (alpha_2 <= 0) & (d_cyl <= distance))[0]
        ts.set_time(f"Find {len(subind_in_cylinder)} cylinder members")

        if use_tree:
            ind_in_cylinder = np.array(
            [ind_candidate[_s] for _s in subind_in_cylinder])
        else:
            ind_in_cylinder = subind_in_cylinder

        d_cyl = d_cyl[sic_3]
        ts.set_time("Build final member list")
        #ts.print_time_usage("_find_cylinder_members")
            
        return ind_in_cylinder, d_cyl

    def _find_cylinder_members(
        self, p1, p2, coords, distance, ifil=None, tree_points=None,
        flag=None, min_dist=None, ind_fil=None):

        ts = TimeStamp()
        
        # Empty return value
        empty = [np.zeros(0, dtype=int), np.zeros(0)]

        pvec = p2 - p1
        pmid = (p1 + p2) / 2
        dp = np.linalg.norm(pvec)
        ts.set_time("Setup")
        if tree_points is None:
            tree_points = cKDTree(coords)
        ind_candidate = tree_points.query_ball_point(
            pmid, np.sqrt((dp/2)**2 + distance**2))
        ts.set_time("Tree query")
        
        # No need to continue if there are no neighbours
        if len(ind_candidate) == 0: return empty
            
        # Check those points in the sphere against the actual cylinder.
        #ind_candidate = np.array(ind_candidate, copy=False)
        ts.set_time("Arrayise ngbs")
        
        q = coords[ind_candidate, :]
        ts.set_time(f"Extract {q.shape[0]} candidate coords")

        alpha_1 = np.dot(q - p1, pvec)
        sic_1 = np.nonzero(alpha_1 >= 0)[0]
        if len(sic_1) == 0: return empty
        alpha_2 = np.dot(q[sic_1, :] - p2, pvec)
        sic_2 = np.nonzero(alpha_2 <= 0)[0]
        if len(sic_2) == 0: return empty
        d_cyl = np.linalg.norm(
            np.cross(q[sic_1[sic_2], :] - p1, pvec), axis=1) / dp 
        sic_3 = np.nonzero(d_cyl <= distance)[0]
        if len(sic_3) == 0: return empty
        subind_in_cylinder = sic_1[sic_2[sic_3]]
        #subind_in_cylinder = np.nonzero(
        #    (alpha_1 >= 0) & (alpha_2 <= 0) & (d_cyl <= distance))[0]
        ts.set_time(f"Find {len(subind_in_cylinder)} cylinder members")

        ind_in_cylinder = np.array(
            [ind_candidate[_s] for _s in subind_in_cylinder])
        #ind_in_cylinder = ind_candidate[subind_in_cylinder]
        d_cyl = d_cyl[sic_3]
        ts.set_time("Build final member list")
        
        # Rest is just for filling output arrays

        if flag is not None:
            flag[ind_in_cylinder] = True
        
        if min_dist is not None:
            subind_best = np.nonzero(
                d_cyl < min_dist[ind_in_cylinder])[0]
            min_dist[ind_in_cylinder[subind_best]] = d_cyl[subind_best]
            ind_fil[ind_in_cylinder[subind_best]] = ifil
        ts.set_time("Fill output arrays")
        #ts.print_time_usage("_find_cylinder_members")
            
        return ind_in_cylinder, d_cyl


        
    def find_filament_lengths(self, store_segment_lengths=False,
        periodic_wrapping=False, use_simple=False, exclude_masked=True):
        """Compute the length of each filament.

        This requires that the coordinates have been converted to Cartesian.
        """
        # First compute the length of each segment, dealing with 2D case
        # in a somewhat crude way.
        if use_simple:
            sampling_coords = self.sampling_data['SimpleCoordinates']
            n_filaments = self.n_filaments_simple
        else:
            sampling_coords = self.sampling_data['CartesianCoordinates']
            n_filaments = self.n_filaments

        sp_x = sampling_coords[:, 0]
        sp_y = sampling_coords[:, 1]
        if self.n_dim == 3:
            sp_z = sampling_coords[:, 2]
        else:
            sp_z = np.zeros_like(sp_x)
        
        # Compute the shortest length of each segment, accounting for possible
        # periodic wrapping if desired
        dx = np.abs(sp_x[1:] - sp_x[:-1])
        dy = np.abs(sp_y[1:] - sp_y[:-1])
        dz = np.abs(sp_z[1:] - sp_z[:-1])
        if periodic_wrapping:
            dx_alt = self.bbox[0, 1] - self.bbox[0, 0] - dx
            dx = np.min(np.vstack((dx, dx_alt)), axis=0)
            dy_alt = self.bbox[1, 1] - self.bbox[1, 0] - dy
            dy = np.min(np.vstack((dy, dy_alt)), axis=0)
            dz_alt = self.bbox[2, 1] - self.bbox[2, 0] - dz
            dz = np.min(np.vstack((dz, dz_alt)), axis=0)
        l_seg = np.sqrt(dx**2 + dy**2 + dz**2)

        # Now iterate through filaments and sum segment lengths
        filament_lengths = np.zeros(n_filaments)
        for ifil in range(n_filaments):
            if use_simple:
                offset = self.filament_data['SamplingPointsOffsetSimple'][ifil]
                end = self.filament_data['SamplingPointsEndSimple'][ifil]
            else:
                offset = self.filament_data['SamplingPointsOffset'][ifil]
                end = self.filament_data['SamplingPointsEnd'][ifil]
            
            l_seg_filament = l_seg[offset:end-1]            
            if exclude_masked:
                if use_simple:
                    mask_bits = (
                        self.sampling_data['SegmentMaskSimple'][offset:end-1])
                else:
                    mask_bits = self.sampling_data['SegmentMask'][offset:end-1]
                l_seg_filament = l_seg_filament[mask_bits == False]
            filament_lengths[ifil] = np.sum(l_seg_filament)

            # For completeness, set the length of the last "segment" in each
            # filament to zero: this is the one connecting the last sampling
            # point to the first one in the next filament (except the last!)
            if ifil < n_filaments-1:
                l_seg[end-1] = 0

        if use_simple:
            self.filament_data['LengthsInMpcSimple'] = filament_lengths
        else:
            self.filament_data['LengthsInMpc'] = filament_lengths
        if store_segment_lengths:
            if use_simple:
                self.sampling_data['SegmentLengthsInMpcSimple'] = l_seg
            else:
                self.sampling_data['SegmentLengthsInMpc'] = l_seg

    def _read_cp_data(self, f, icp):
        """Read data for one critical point from the ASCII file.

        Parameters
        ----------
        f : object
            The file handle to read from
        icp : int
            The index (starting from 0) of the point to read. This is only for
            recording, there is no way to read non-sequentially.

        Returns
        -------
        cp_struct : tuple with read-in data.
        """

        # First part is relatively easy: the intrinsic properties
        line = f.readline()
        numbers = numbers_from_string(line)
        if len(numbers) != self.n_dim + 4:
            raise ValueError(
                f"Expected to read {self.n_dim + 4} numbers for critical point"
                f"{icp}, but got {len(numbers)} instead!"
            )
        data = {}
        data['Type'] = numbers[0]
        data['Coordinates'] = np.array(numbers[1:self.n_dim+1])
        data['Value'] = numbers[self.n_dim + 1]
        data['PairID'] = numbers[self.n_dim + 2]
        data['BoundaryFlag'] = numbers[self.n_dim + 3]

        # Now the connection to filaments...
        data['NumberOfFilaments'] = int(f.readline())
        data['Filaments'] = [(None, None)] * data['NumberOfFilaments']
        for ifil in range(data['NumberOfFilaments']):
            line = f.readline()
            numbers = numbers_from_string(line)
            if len(numbers) != 2:
                raise ValueError(
                    f"Expected exactly two numbers per filament, but got "
                    f"{len(numbers)} (filament {ifil}, critical point {icp})!"
                )
            data['Filaments'][ifil] = tuple(numbers)

        return data

    def _read_filament_data(self, f, ifil):
        """Read data for one filament from the ASCII file.

        Parameters
        ----------
        f : object
            The file handle to read from
        ifil : int
            The index (starting from 0) of the filament to read. This is purely
            for recording, there is no way to read non-sequentially.

        Returns
        -------
        data : dict
            Structure holding the data for this filament.
        """
        # First line is straightforward (3 numbers)
        line = f.readline()
        numbers = numbers_from_string(line)
        if len(numbers) != 3:
            raise ValueError(
                f"Expected to read 3 numbers for filament {ifil}, but got "
                f"{len(numbers)} instead!"
            )
        data = {}
        data['CriticalPoints'] = np.array(numbers[:2])
        data['NumberOfSamplingPoints'] = numbers[2]
        data['SamplingPoints'] = np.zeros(
            (data['NumberOfSamplingPoints'], self.n_dim))

        for isample in range(data['NumberOfSamplingPoints']):
            line = f.readline()
            numbers = numbers_from_string(line)
            if len(numbers) != self.n_dim:
                raise ValueError(
                    f"Expected exactly {self.n_dim} numbers per sampling "
                    f"point, but got {len(numbers)} instead "
                    f"(filament {ifil}, sampling point {isample})!"
                )
            data['SamplingPoints'][isample] = tuple(numbers)

        return data


    def _initialize_cp_structure(self):
        """Initialize the internal cp_data structure."""
        self.cp_data = {}
        self.cp_data['Type'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['Coordinates'] = np.zeros((self.n_cp, self.n_dim)) - 1
        self.cp_data['Value'] = np.zeros(self.n_cp) - 1
        self.cp_data['PairID'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['BoundaryFlag'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['NumberOfFilaments'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['CPFilamentsOffset'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['CPFilamentsEnd'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['PersistenceRatio'] = np.zeros(self.n_cp) - 1
        self.cp_data['PersistenceNSigmas'] = np.zeros(self.n_cp) - 1
        self.cp_data['Persistence'] = np.zeros(self.n_cp) - 1
        self.cp_data['PersistencePair'] = np.zeros(self.n_cp) - 1
        self.cp_data['Robustness'] = np.zeros(self.n_cp) - 1
        self.cp_data['RobustnessRatio'] = np.zeros(self.n_cp) - 1
        self.cp_data['ParentIndex'] = np.zeros(self.n_cp, dtype=int) - 1
        self.cp_data['ParentLogIndex'] = np.zeros(self.n_cp) - 1
        self.cp_data['LogFieldValue'] = np.zeros(self.n_cp) - 1
        self.cp_data['FieldValue'] = np.zeros(self.n_cp) - 1
        self.cp_data['Cell'] = np.zeros(self.n_cp) - 1

    def _initialize_filament_structure(self):
        """Initialize the internal structure for filament data."""
        self.filament_data = {}
        self.filament_data['CriticalPoints'] = np.zeros(
            (self.n_filaments, 2), dtype=int) - 1
        self.filament_data['NumberOfSamplingPoints'] = np.zeros(
            self.n_filaments, dtype=int) - 1
        self.filament_data['SamplingPointsOffset'] = np.zeros(
            self.n_filaments, dtype=int) - 1
        self.filament_data['SamplingPointsEnd'] = np.zeros(
            self.n_filaments, dtype=int) - 1

    def _initialize_filament_sample_structure(self):
        """Initialize the internal structure for filament sampling points."""
        self.sampling_data = {}
        self.sampling_data['Coordinates'] = FlexArray(
            (self.n_filaments*10, self.n_dim), dtype=float)

    def _add_filament_sample_fields(self):
        """Add secondary fields to dict, once we know the number of samples."""
        self.sampling_data['FieldValue'] = np.zeros(self.n_sample) - 1
        self.sampling_data['Orientation'] = np.zeros(self.n_sample) - 1
        self.sampling_data['Cell'] = np.zeros(self.n_sample) - 1
        self.sampling_data['LogFieldValue'] = np.zeros(self.n_sample) - 1
        self.sampling_data['Type'] = np.zeros(self.n_sample, dtype=int) - 1
        self.sampling_data['Robustness'] = np.zeros(self.n_sample) - 1
        self.sampling_data['RobustnessRatio'] = np.zeros(self.n_sample) - 1


class FlexArray:
    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype) - 1
        self.allocated_size = shape[0]
        self.next_index = 0

    def append(self, data):
        if self.next_index >= self.data.shape[0]:
            old_length = self.data.shape[0]
            new_length = int(old_length * 1.5)
            new_shape = list(self.data.shape)
            new_shape[0] = new_length
            self.data.resize(new_shape)
            self.data[old_length:, ...] = -1

        self.data[self.next_index, ...] = data
        self.next_index += 1

    def shrink(self):
        self.data = self.data[:self.next_index, ...]
        self.allocated_size = self.next_index


def numbers_from_string(string):
    #rr = re.findall(
    #    "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
    rr = re.findall('[\d]*[.][\d]*[eE][-+][\d]+|[\d]*[eE][-+][\d]+|[\d]*[.][\d]+|[\d]+', string)
    numbers = []
    for ir in rr:
        try:
            inum = int(ir)
        except ValueError:
            inum = float(ir)
        numbers.append(inum)
    return numbers


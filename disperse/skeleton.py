"""Skeleton class to represent a DisPerSE skeleton instance."""

from pdb import set_trace
import re
import numpy as np

class Skeleton:
	"""Representation of a DisPerSE skeleton.

	Parameters
	----------
	filename : str
		The name of the .a.NDskl file from DisPerSE that contains the
		skeleton data.
	"""
	def __init__(self, filename):
		with open(filename, 'r') as f:

			# The first line is just the header
			line = f.readline()
			if line != 'ANDSKEL\n':
				raise ValueError("Invalid input file.")

			# Read number of dimensions
			self.n_dim = int(f.readline())
			print(f"Instantiating skeleton with {self.n_dim} dimensions...")

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
			
			# Critical points...
			line = f.readline()
			if line != '[CRITICAL POINTS]\n':
				raise ValueError(
					f"Expected to read '[CRITICAL POINTS]', got '{line}'!")
			self.n_cp = int(f.readline())
			print(f"Expecting {self.n_cp} critical points...")

			# Now that we know how many Critical Points to expect, we can 
			# initialize the data structure that holds all their information
			self._initialize_cp_structure()

			# We also need to initialize the list that holds all the 
			# critical point --> filament links:
			self.cp_filaments = np.zeros((self.n_cp * 4, 2), dtype=int) - 1
			self.cp_filament_entries = 0

			# Read and process data block for each critical point in turn...
			for icp in range(self.n_cp):
				if icp % 1000 == 0:
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
				curr_index = self.cp_filament_entries
				cp_nfil = cp_dict['NumberOfFilaments']
				self.cp_data['CPFilamentsOffset'][icp] = curr_index
				self.cp_data['CPFilamentsEnd'] = curr_index + cp_nfil

				# ... and now put the data there
				for ifil in range(cp_dict['NumberOfFilaments']):
					if self.cp_filament_entries >= self.cp_filaments.shape[0]:
						old_length = self.cp_filaments.shape[0]
						self.cp_filaments.resize(
							(int(self.cp_filaments.shape[0]*1.5), 2))
						self.cp_filaments[old_length:, :] = -1

					self.cp_filaments[self.cp_filament_entries, :] = (
							cp_dict['Filaments'][ifil])
					self.cp_filament_entries += 1

			# We can now truncate `self.cp_filaments` to the actual size
			self.cp_filaments = (
				self.cp_filaments[:self.cp_filament_entries, ...])

			# Yay! Done with critical points (for now...). Next up: filaments
			line = f.readline()
			if line != '[FILAMENTS]\n':
				raise ValueError(
					f"Expected to read '[FILAMENTS]', got '{line}'! ")
			
			# Read the number of filaments. This must be exactly half the
			# number of critical point - filament connections read above,
			# because each filament connects exactly two CPs. If not -- bad.
			self.n_filaments = int(f.readline())
			if self.n_filaments != self.cp_filament_entries / 2:
				raise ValueError(
					f"Read {self.cp_filament_entries} CP-filament "
					f"connections, but there are {self.n_filaments} "
					f"filaments. This does not add up!"
				)

			# Initialize the data structures for filaments and their sampling
			# points
			self._initialize_filament_structure()
			self._initialize_filament_sample_structure()

			# Read filaments one-by-one
			for ifil in range(self.n_filaments):
				if ifil % 1000 == 0:
					print(f"Loading Filament {ifil}...")
				fil_dict = self._read_filament_data(f, ifil)

				# Now we have to unpack these values...
				for field in ['CriticalPoints', 'NumberOfSamplingPoints']:
					self.filament_data[field][ifil, ...] = fil_dict[field]

				# First record where in the full sampling point list this
				# filament's data will be stored...
				self.filament_data['SamplingPointsOffset'][ifil] = (
					self.sampling_data['NextIndex'])
				self.filament_data['SamplingPointsEnd'][ifil] = (
					self.sampling_data['NextIndex'] +
					fil_dict['NumberOfSamplingPoints']
				)

				# ... and now get the data there
				for isample in range(fil_dict['NumberOfSamplingPoints']):
					index = self.sampling_data['NextIndex']
					if index >= self.sampling_data['AllocatedSize']:
						old_length = self.sampling_data['Coordinates'].shape[0]
						self.sampling_data['Coordinates'].resize(
							(int(self.sampling_data.shape[0]*1.5), self.n_dim))
						self.sampling_data['Coordinates'][old_length:, :] = -1

					self.sampling_data['Coordinates'][index, :] = (
						fil_dict['SamplingPoints'][isample])
					self.sampling_data['NextIndex'] += 1
				
			# We can now truncate the sampling coordinates to actual size
			n_sample = self.sampling_data['NextIndex']
			self.sampling_data['Coordinates'] = (
				self.sampling_data['Coordinates'][:n_sample, ...])
			self.sampling_data['AllocatedSize'] = n_sample
			self.n_sample = n_sample

			# ----- Done reading basic data. Now comes ancillary info... ----

			print("Loading extra info for critical points...")

			line = f.readline()
			if line != '[CRITICAL POINTS DATA]\n':
				raise ValueError(
					f"Expected to read '[CRITICAL POINTS DATA]', got "
					f"'{line}'!"
				)
			self.n_cp_extra_fields = int(f.readline())
			if self.n_cp_extra_fields != 9:
				raise ValueError(
					f"Expected to read 9 extra fields for critical points, "
					f"not {self.n_cp_extra_field}!"
				)
			field_names = [
				'persistence_ratio', 'persistence_nsigmas', 'persistence',
				'persistence_pair', 'parent_index', 'parent_log_index',
				'log_field_value', 'field_value', 'cell'
			]
			field_names_internal = [
				'PersistenceRatio', 'PersistenceNSigmas', 'Persistence',
				'PersistencePair', 'ParentIndex', 'ParentLogIndex',
				'LogFieldValue', 'FieldValue', 'Cell'
			]

			for field in field_names:
				line = f.readline()
				if line != f'{field}\n':
					raise ValueError(
						f"Expected field name '{field}', not '{line}'!")

			# Read extra CP info, line by line...
			for icp in range(self.n_cp):
				numbers = numbers_from_string(f.readline())
				for ifield, field in enumerate(field_names_internal):
					self.cp_data[field][icp] = numbers[ifield]

			# And finally, filament sampling points ancillary info
			print("Loading extra info for filament sampling points...")
			line = f.readline()
			if line != '[FILAMENTS DATA]\n':
				raise ValueError(
					f"Expected to read '[FILAMENTS DATA]', not '{line}'!")

			self.n_sample_extra_fields = int(f.readline())
			if self.n_sample_extra_fields != 5:
				raise ValueError(
					f"Expected to read 5 extra fields for filaments sampling "
					f"points, not {self.n_sample_extra_fields}!"
				)
			field_names = [
				'field_value', 'orientation', 'cell', 'log_field_value',
				'type'
			]
			field_names_internal = [
				'FieldValue', 'Orientation', 'Cell', 'LogFieldValue', 'Type']
			for field in field_names:
				line = f.readline()
				if line != f'{field}\n':
					raise ValueError(
						f"Expected field name '{field}', not '{line}'!")

			# Read sampling point info, line by line...
			self._add_filament_sample_fields()
			for isample in range(self.n_sample):
				numbers = numbers_from_string(f.readline())
				for ifield, field in enumerate(field_names_internal):
					self.sampling_data[field][isample] = numbers[ifield]


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
		self.sampling_data['AllocatedSize'] = self.n_filaments * 10
		self.sampling_data['NextIndex'] = 0
		self.sampling_data['Coordinates'] = np.zeros(
			(self.n_filaments * 10, 3)) - 1

	def _add_filament_sample_fields(self):
		"""Add secondary fields to dict, once we know the number of samples."""
		self.sampling_data['FieldValue'] = np.zeros(self.n_sample) - 1
		self.sampling_data['Orientation'] = np.zeros(self.n_sample) - 1
		self.sampling_data['Cell'] = np.zeros(self.n_sample) - 1
		self.sampling_data['LogFieldValue'] = np.zeros(self.n_sample) - 1
		self.sampling_data['Type'] = np.zeros(self.n_sample, dtype=int) - 1


def numbers_from_string(string):
	rr = re.findall(
		"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
	numbers = []
	for ir in rr:
		try:
			inum = int(ir)
		except ValueError:
			inum = float(ir)
		numbers.append(inum)
	return numbers


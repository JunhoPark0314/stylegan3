import copy
import os

from yacs.config import CfgNode as CN
from yacs.config import _assert_with_logging, _check_and_coerce_cfg_value_type

is_replace = lambda x: (hasattr(x, 'replace') and x.replace)

def _merge_a_into_b_wi_replace(a, b, root, key_list):
	_assert_with_logging(
		isinstance(a, CfgNode),
		"`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
	)
	_assert_with_logging(
		isinstance(b, CfgNode),
		"`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
	)

	for k, v_ in a.items():
		full_key = ".".join(key_list + [k])

		v = copy.deepcopy(v_)
		v = b._decode_cfg_value(v)

		if k in b:
			v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
			# Recursively merge dicts
			if (isinstance(v, CfgNode) is False) or is_replace(v):
				b[k] = v
			elif isinstance(v, CfgNode):
				try:
					_merge_a_into_b_wi_replace(v, b[k], root, key_list + [k])
				except BaseException:
					raise
		elif b.is_new_allowed():
			b[k] = v
		else:
			if root.key_is_deprecated(full_key):
				continue
			elif root.key_is_renamed(full_key):
				root.raise_key_rename_error(full_key)
			else:
				raise KeyError("Non-existent config key: {}".format(full_key))

def clean_replace(cfg):
	if hasattr(cfg, 'replace'):
		cfg.__delitem__('replace')
	else:
		for k, v in cfg.items():
			if isinstance(v, CfgNode):
				clean_replace(v)


class CfgNode(CN):
	# def __init__(self,init_dict=None, key_list=None, new_allowed=False):
	# 	super().__init__(init_dict, key_list, new_allowed)

	def merge_from_other_cfg(self, cfg_other):
		"""Merge `cfg_other` into this CfgNode."""
		_merge_a_into_b_wi_replace(cfg_other, self, self, [])
	
	def merge_from_file(self, cfg_filename):
		"""Load a yaml config file and merge it this CfgNode."""
		with open(cfg_filename, "r") as f:
			cfg = self.load_cfg(f)

		if hasattr(cfg, '__BASE__'):
			dir_name = os.path.dirname(cfg_filename)
			for basefile in cfg.__BASE__:
				self.merge_from_file(os.path.join(dir_name, basefile))
			cfg.__delitem__('__BASE__')

		self.merge_from_other_cfg(cfg)
	
	def clear_build(self):
		self.clean_replace()
		self.calc_lambda(None, self)
	
	def clean_replace(self):
		clean_replace(self)

	def calc_lambda(self, key, root):
		if not isinstance(root, CfgNode):
			return root if root != 'None' else None
		elif hasattr(root, 'inline'):
			lambda_args = {k: getattr(self, root[k]) for k in root.keys() if k != 'inline'}
			result = eval(root.inline)(**lambda_args)
			return result
		else:
			key_format = '{par}.{child}' if key is not None else '{child}'
			for k in root.keys():
				v = getattr(root, k)
				if isinstance(v, CfgNode):
					setattr(root, k, self.calc_lambda(key_format.format(par=key, child=k),v))
			return root
	
	def __getattr__(self, __k):
		if not isinstance(__k, str):
			return super().__getattr__(__k)
		__klist  = __k.split('.', 1)
		if len(__klist) == 1:
			return super().__getattr__(__klist[0])
		else:
			return super().__getattr__(__klist[0]).__getattr__(__klist[1])
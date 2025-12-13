# domain/adapters/__init__.py
"""
Adapters do domínio para comunicação com sistemas externos.
"""

from .findface_adapter import FindfaceAdapter

__all__ = ['FindfaceAdapter']
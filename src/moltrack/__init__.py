__all__ = (
    "MolTrack",
)

__version__ = "2.0.0"


def __getattr__(name):
    # Scientific adapters can be used in workers without loading Qt or napari.
    if name == "MolTrack":
        from ._widget import MolTrack

        return MolTrack
    raise AttributeError(name)

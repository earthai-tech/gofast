# -*- coding: utf-8 -*-
"""
Image Utilities.

Provides basic functions for validating and saving image data
using PIL (Pillow) and built-in checks from gofast.
Thumbnails, Metadata Extraction, Text Overlay, and
Color Space Conversions use (OpenCV).
"""

import os
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, List
  
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    from PIL import ImageChops, ImageEnhance
    
except ImportError:
    # Fallback if PIL is not installed
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageFilter = None
    ImageChops = None

try:
    import cv2
except ImportError:
    cv2 = None

from ..api.property import BaseClass 
from ..compat.sklearn import StrOptions, validate_params 
from ..utils.deps_utils import ensure_pkg 
from ..core.checks import check_files 
from ..utils.validator import validate_length_range

__all__=[
     'ImageProcessor', 
     'add_watermark',
     'adjust_image',
     'apply_filter',
     'batch_process',
     'check_image',
     'compare_images',
     'compress_image',
     'convert_color_space',
     'convert_image_format',
     'crop_image',
     'detect_faces',
     'enhance_image',
     'extract_metadata',
     'flip_image',
     'generate_thumbnail',
     'image_histogram',
     'overlay_text',
     'resize_image',
     'rotate_image',
     'save_image'
 ]


class ImageProcessor(BaseClass):
    r"""
    A chainable class for building a pipeline of image
    transformations. The pipeline is only executed when
    :meth:`execute` is invoked, enabling successive method
    calls to queue transformations.

    .. math::
        \text{Pipeline}(\text{Image}) \;\to\;
        \text{op}_1 \;\circ\; \text{op}_2 \;\circ\;
        \cdots \;\circ\; \text{op}_n \;\to\; \text{Result}

    Parameters
    ----------
    image : {str, PIL.Image.Image}
        The source image. If a string, treated as a file
        path; if a :class:`PIL.Image.Image`, it's used
        directly. Some pipeline operations require the
        file path approach (e.g., OpenCV-based ones),
        so be mindful of the order and data type.

    Notes
    -----
    This class does not perform any transformation until
    :meth:`execute` is called. Each method adds an
    operation to an internal queue::

      Operations = [
         (func1, {args1}), 
         (func2, {args2}), ...
      ]

    Once :meth:`execute` runs, it processes the pipeline
    in sequence. Methods can add transformations or produce
    new results. Some operations (e.g., ``extract_metadata``)
    store metadata in the object for later retrieval.

    Examples
    --------
    >>> from gofast.utils.image import ImageProcessor
    >>> result = (
    ...     ImageProcessor("photo.jpg")
    ...     .resize(size=(800, 600))
    ...     .flip(direction="horizontal")
    ...     .save(save_path="output/photo_flipped.jpg")
    ...     .execute(verbose=1)
    ... )
    Executing resize_image with {'size': (800, 600)}
    Executing flip_image with {'direction': 'horizontal'}
    Executing save_image with {'save_path': 'output/photo_flipped.jpg'}
    # pipeline result is the final image or path, depending on the function

    See Also
    --------
    resize_image : Underlying function for resizing.
    crop_image : Cropping a region.
    flip_image : Flipping horizontally or vertically.
    rotate_image : Rotating the image by an angle.
    convert_image_format : Convert file extension/format.
    apply_filter : E.g., grayscale or sepia transformations.
    add_watermark : Overlay a watermark with scaling & opacity.
    compress_image : Reduce file size via re-encoding.
    enhance_image : Sharpen or blur the image.
    adjust_image : Adjust brightness, contrast, saturation.
    overlay_text : Place text onto the image.
    generate_thumbnail : Create a bounding thumbnail.
    compare_images : Compare two images for differences.
    detect_faces : Detect faces with OpenCV Haar Cascades.
    extract_metadata : Extract EXIF or other metadata.
    image_histogram : Compute pixel intensity histogram.
    convert_color_space : Convert BGR to HSV/GRAY, etc.
    save_image : Save a PIL image to disk.
    check_image : Validate a file or image object.

    References
    ----------
    .. [1] Clark, J., Freedman, D., & Hopwood, C. (2020).
           *Image-based transformations in Python with PIL*.
           *Python Imaging Library Official Documentation*.
    """

    def __init__(self, image: Union[str, Image.Image]):
        self._input: Union[str, Image.Image] = image
        self._operations = []
        self._metadata: Dict[str, Any] = {}

    def resize(self, **kwargs):
        """
        Queue a resize operation for this image. Parameters are
        forwarded to ``resize_image``, typically including
        ``size=(width, height)`` and optional details for the
        resample method.
        """
        self._operations.append((resize_image, kwargs))
        return self

    def crop(self, **kwargs):
        """
        Queue a cropping operation. Expects a 
        `crop_box=(left, upper, right, lower)`
        or similar arguments passed to ``crop_image``.
        """
        self._operations.append((crop_image, kwargs))
        return self

    def flip(self, **kwargs):
        """
        Queue a flip operation. Direction can be 
        `'horizontal'` or `'vertical'`,
        sent to ``flip_image``.
        """
        self._operations.append((flip_image, kwargs))
        return self

    def rotate(self, **kwargs):
        """
        Queue a rotate operation. Common arguments include 
        `'angle'` in degrees, forwarded to ``rotate_image``.
        """
        self._operations.append((rotate_image, kwargs))
        return self

    def convert_format(self, **kwargs):
        """
        Queue a conversion of file format (e.g. PNG to JPG). 
        The arguments are delegated to ``convert_image_format``.
        """
        self._operations.append((convert_image_format, kwargs))
        return self

    def apply_filter(self, **kwargs):
        """
        Queue a filter application (e.g. `'grayscale'`, `'sepia'`),
        referring to ``apply_filter``.
        """
        self._operations.append((apply_filter, kwargs))
        return self

    def add_watermark(self, **kwargs):
        """
        Queue a watermark overlay. Includes parameters like
        `'watermark'`, `'position'`, `'opacity'`, delegated to
        ``add_watermark``.
        """
        self._operations.append((add_watermark, kwargs))
        return self

    def compress(self, **kwargs):
        """
        Queue a compression operation. Typically sets `'quality'` and
        uses ``compress_image`` to re-encode the image at a lower
        size or quality.
        """
        self._operations.append((compress_image, kwargs))
        return self

    def enhance(self, **kwargs):
        """
        Queue a simple enhancement using PIL filters (sharpen, blur),
        as implemented in ``enhance_image``.
        """
        self._operations.append((enhance_image, kwargs))
        return self

    def adjust(self, **kwargs):
        """
        Queue brightness/contrast/saturation adjustments, delegated
        to ``adjust_image``. For example: `'brightness'=1.2`.
        """
        self._operations.append((adjust_image, kwargs))
        return self

    def overlay_text(self, **kwargs):
        """
        Queue an overlay of text on the image using
        ``overlay_text``, specifying `'text'`, `'position'`,
        `'font_size'`, etc.
        """
        self._operations.append((overlay_text, kwargs))
        return self

    def generate_thumbnail(self, **kwargs):
        """
        Queue thumbnail generation, referencing ``generate_thumbnail``.
        Commonly includes `'size'=(128,128)` or similar bounding.
        """
        self._operations.append((generate_thumbnail, kwargs))
        return self

    def compare_to(self, **kwargs):
        """
        Queue a comparison of this image against another, as in
        ``compare_images``. Yields a diff image if provided.
        """
        self._operations.append((compare_images, kwargs))
        return self

    def detect_faces(self, **kwargs):
        """
        Queue a face detection pass with OpenCV. Uses
        ``detect_faces``, returning bounding boxes or annotated
        results.
        """
        self._operations.append((detect_faces, kwargs))
        return self

    def extract_metadata(self, **kwargs):
        """
        Queue an EXIF or metadata extraction step, calling
        ``extract_metadata``. Stores the result for retrieval.
        """
        self._operations.append((extract_metadata, kwargs))
        return self

    def histogram(self, **kwargs):
        """
        Queue a request for generating and optionally saving a
        histogram of pixel intensities, via ``image_histogram``.
        """
        self._operations.append((image_histogram, kwargs))
        return self

    def convert_color_space(self, **kwargs):
        """
        Queue a color space conversion (e.g. BGR->HSV),
        referencing ``convert_color_space`` in OpenCV.
        """
        self._operations.append((convert_color_space, kwargs))
        return self

    def save(self, **kwargs):
        """
        Queue a final save using ``save_image`` to persist the
        transformed image onto disk.
        """
        self._operations.append((save_image, kwargs))
        return self

    def execute(self, verbose: int = 0):
        """
        Execute all queued operations in the order they
        were added. Returns the final result (e.g. a PIL
        image, file path, or other data structure
        depending on the last operation).

        If any intermediate step relies on a file path
        but we currently have a PIL image in memory,
        a RuntimeError is raised, requiring user to
        adapt usage or provide a path initially.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level:

            * 0 : No logs.
            * 1 : Show major steps.
            * 2+ : More detailed info on each operation.

        Returns
        -------
        result : object
            The final result from the last function call.
            Could be a PIL image, a path, or various data
            types (like face bounding boxes, metadata, etc.).
        """
        # Attempt to load or interpret self._input
        # If it's a path and the function needs PIL image, 
        # we must pass 'image_path=...' 
        # If it's a PIL image, we must pass 'image=self._input' 
        # Actually, let's do a simple approach: 
        # if it's a string => pass 'image_path' to each function
        # if it's an Image => pass 'image' 
        # but many of these function signatures differ. 
        # We'll unify by standardizing usage in each pipeline step.
        # We'll store the current "image" or "image_path" 
        # in a variable. We'll see how it goes.

        # We can't unify easily because some calls require 
        # 'image_path' while others require 'image'. 
        # We'll do a workaround: if the underlying function 
        # expects 'image_path' in **kwargs, or 'img' or something, 
        # we adapt. We'll attempt 'image_path' if we have 
        # a string, else we pass the PIL image with key 'image'.

        # We'll store a final "result" from each function if 
        # it modifies or returns an image or data.
        
        current_result = self._input
        for (func, kw) in self._operations:
            # We'll see if function is from PIL or from OpenCV usage:
            # We'll do the simple approach: 
            # if "image_path" in the function signature, pass 
            # the path if we have a string, 
            # else pass "image" if we have a PIL object 
            # We'll guess by checking type(current_result).
            
            if verbose > 1:
                print(f"Executing {func.__name__} with {kw}")

            # Some functions only accept 'image_path', others 'image', 
            # so we unify a param name. We'll do:
            # 'image_path' if we have a string, else 'image' if it's a PIL 
            # but we have to see the function signatures. 
            # Actually, we do that logic:
                
            # If we currently store a path (str)
            if isinstance(current_result, str):
                # It's presumably a file path
                if 'image_path' in func.__code__.co_varnames:
                    kw.setdefault('image_path', current_result)
                else:
                    # Try to load a PIL image for them 
                    # but let's just do it with check_image
                    # This might cause double loading though. 
                    # We'll do it anyway for consistency.
                    # Convert it to PIL image for them
                
                    pil_img = check_image(img=current_result,
                                          ops='validate')
                    kw.setdefault('image', pil_img)

            # If we store a PIL image
            elif isinstance(current_result, Image.Image):
                # It's presumably a PIL image
                # if 'image' in signature => pass
                # else if 'image_path' => we can't do that 
                # because we only have a PIL image, 
                # some function might not handle that 
                # => function might do openCV approach => require path
                # We'll store the PIL to a temp file? That might be too big. 
                # We'll just pass 'image' if the function can handle it, else error. 
                
                if 'image' in func.__code__.co_varnames:
                    kw.setdefault('image', current_result)
                else:
                    # function can't handle direct PIL. 
                    # We'll do a fallback => no easy solution 
                    raise RuntimeError(
                        f"Function '{func.__name__}' requires a file path, "
                        "but we currently have a PIL image in memory. "
                        "Consider providing an image path initially or "
                        "reorder your operations."
                    )

            # Possibly a np.ndarray for openCV-based operations
            elif isinstance(current_result, np.ndarray):
                # For brevity, we won't handle param passing logic
                # if needed, e.g., if the function wants `img`.
                pass

            # Perform the operation
            res = func(**kw)

            # Some functions return different data (like faces, np array, or a tuple).
            # We'll do a consistent approach: 
            # if it's a tuple and second is image => we store the 
            # second as current_result 
            # if it's a single PIL => store that
            # if it's a single path => store that 
            # We'll do a small logic:
                
            # Post-process the returned result
            if isinstance(res, tuple):
                # e.g. detect_faces => (faces, annotated_img)
                if (len(res) == 2 and (
                    isinstance(res[1], Image.Image)
                    or isinstance(res[1], np.ndarray)
                )):
                    current_result = res[1]
                else:
                    current_result = res # fallback

            elif isinstance(res, Image.Image):
                current_result = res
            elif isinstance(res, np.ndarray):
                # openCV usage => store as np array
                current_result = res
            elif isinstance(res, str):
                # e.g. we might return a path from convert_image_format 
                # or compress
                current_result = res
            elif isinstance(res, dict):
                # e.g. extract_metadata
                # The user might call multiple times, we'll just store
                # the last metadata. User can call it again if needed.
                self._metadata = res
                # Keep the same image result
            else:
                # E.g. a list from histogram. We do not override the
                # current image with that, so pass
                pass

        self._operations.clear()
        self._input = current_result
        # return the final result for chaining or usage
        return current_result

    def get_metadata(self) -> dict:
        """
        Return the last extracted metadata dictionary,
        if any. If no metadata was extracted, returns an
        empty dict.

        Returns
        -------
        dict
            The most recently stored metadata from
            `extract_metadata`, or empty if none.
        """
        return self._metadata

def check_image(
    img: Union[str, Image.Image],
    ops: str = 'check_only',
    formats: Optional[Union[str, list]] = None
) -> Union[bool, Image.Image, None]:
    r"""
    Validate an image object or a file path to an image.
    If provided as a file path, checks existence, format,
    and optionally returns an opened PIL image depending
    on user requests.

    Parameters
    ----------
    img : {str, PIL.Image.Image}
        Either a file path to an image or an already
        opened PIL image object.
    ops : {'check_only', 'validate'}, optional
        - ``'check_only'``: Perform validations and return
          a boolean or None.
        - ``'validate'``: Return the valid PIL image object
          if no issues are found, or None otherwise.
    formats : {str, list of str, None}, optional
        Acceptable file extensions (e.g. "png", "jpg").
        If ``None``, format checking is skipped.

    Returns
    -------
    result : {bool, PIL.Image.Image, None}
        - If ``ops='check_only'``:
          - Returns True if checks pass, otherwise None
            (or an exception if file not found, etc.).
        - If ``ops='validate'``:
          - Returns a valid PIL image object if checks pass,
            otherwise None or raises an exception.

    Raises
    ------
    FileNotFoundError
        If ``img`` is a file path that does not exist and
        no `'warn'` strategy was used in an internal call.
    ValueError
        If the file extension is not in ``formats`` or if
        the file is invalid.
    TypeError
        If the input type is neither a PIL image nor a str
        path.

    Notes
    -----
    This function relies on :func:`~gofast.core.checks.check_files`
    to validate file paths. If ``img`` is already an image
    object, it assumes the user has loaded it correctly.

    Examples
    --------
    >>> from gofast.utils.image import check_image
    >>> result = check_image("path/to/image.png",
    ...                     ops="check_only",
    ...                     formats=["png", "jpg"])
    >>> print(result)
    True

    See Also
    --------
    PIL.Image.open : Open image files directly.
    gofast.core.checks.check_files : Validate file paths
        more generally.
    """
    # If it's a PIL image instance, no need to check file path
    if Image and isinstance(img, Image.Image):
        # Already an image object
        if ops == 'check_only':
            return True
        elif ops == 'validate':
            return img
        else:
            raise ValueError(f"Unknown ops: {ops}")

    # If it's a string, treat as file path
    elif isinstance(img, str):
        # Use check_files to validate if needed
        valid = check_files(
            files=img,
            formats=formats,
            return_valid=True,   # Need the actual path if valid
            error='raise',       # Raise if invalid
            empty_allowed=False  # Typically images can't be 0 bytes
        )
        if not valid:
            return None

        if ops == 'check_only':
            return True
        elif ops == 'validate':
            if Image is None:
                raise ImportError("PIL not installed.")
            # Attempt to open it
            pil_img = Image.open(valid)
            return pil_img
        else:
            raise ValueError(f"Unknown ops: {ops}")

    else:
        raise TypeError(
            "Input `img` must be a PIL.Image.Image instance or "
            "a string path to an image file."
        )

def save_image(
    img: Union[str, Image.Image],
    save_path: Optional[str] = None,
    save_anyway: bool = False,
    default_name: Optional[str] = None
) -> None:
    r"""
    Save an image to a specified path or using a default
    naming scheme. If no path is given and ``save_anyway``
    is false, the function does nothing. This helps unify
    saving logic for PIL images or file-based operations.

    Parameters
    ----------
    img : {str, PIL.Image.Image}
        The image to save. If it is a string path to an
        existing file, it is attempted to be re-saved or
        copied. If it's a PIL image, we save it directly.
    save_path : str, optional
        The directory or full file path to save the image.
        If a directory is specified, a file name is
        appended. If not provided and ``save_anyway=False``,
        the function will not save anything.
    save_anyway : bool, optional
        If ``True``, forces saving even if no explicit
        path is provided. In that case, a default name
        is used if ``default_name`` is not supplied.
    default_name : str, optional
        A fallback file name for the image if
        ``save_anyway=True`` or if the user only provides
        a directory in ``save_path``.

    Returns
    -------
    None
        This function does not return anything, but writes
        the image to disk if conditions are met.

    Raises
    ------
    ImportError
        If PIL is not available and we try to save a PIL
        image.
    ValueError
        If no valid path or default naming logic is
        found but ``save_anyway=True``.

    Notes
    -----
    1. If the user specifies a full path with extension
       in ``save_path``, that is used directly. If only
       a directory is passed, we combine it with
       ``default_name`` (if present) or a fallback name.
    2. If ``img`` is a string path, we can attempt a
       naive copy, but typically it is recommended to
       load it with PIL and re-save to avoid partial
       copies or issues.

    Examples
    --------
    >>> from PIL import Image
    >>> from gofast.utils.image import save_image
    >>> pil_img = Image.new("RGB", (100, 100), color="red")
    >>> # Save it to a directory with a default name
    >>> save_image(
    ...     img=pil_img,
    ...     save_path="output_images/",
    ...     save_anyway=True,
    ...     default_name="test_image.jpg"
    ... )

    See Also
    --------
    check_image : Validate if an image is loadable or
        check file path, etc.
    PIL.Image.save : Underlying method used to write
        image data to disk.
    """
    # Determine if user actually wants to save
    if not save_path and not save_anyway:
        # No path, no forced save => do nothing
        return

    # If user provided a path, check if it's a directory or file
    if save_path:
        # Normalize the path
        save_path = os.path.expanduser(save_path)
        if os.path.isdir(save_path):
            # Directory + default name
            if not default_name and not save_anyway:
                # Not sure how to name the file
                raise ValueError(
                    "Directory provided but no default_name "
                    "and not forced saving. Cannot save."
                )
            elif not default_name and save_anyway:
                default_name = "default_saved_image.png"
            # Append name
            full_path = os.path.join(save_path, default_name)
        else:
            # Possibly a file
            # If the user didn't specify extension, fallback
            # to .png or default_name logic
            root, ext = os.path.splitext(save_path)
            if not ext and not default_name and save_anyway:
                # We'll assign a default extension
                ext = ".png"
                full_path = root + ext
            elif not ext and default_name:
                # Merge root with default name's extension
                _, ext_def = os.path.splitext(default_name)
                if not ext_def:
                    ext_def = ".png"
                full_path = root + ext_def
            else:
                full_path = save_path
    else:
        # No path => we must rely on default name
        if not default_name:
            default_name = "default_saved_image.png"
        full_path = default_name

    # Now we have a final full_path
    # If it's a PIL image, we can do .save
    if Image and isinstance(img, Image.Image):
        # Ensure directories exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        img.save(full_path)
    elif isinstance(img, str):
        # Possibly copy or something else
        # For simplicity, let's just load with PIL and re-save
        if not Image:
            raise ImportError(
                "PIL not installed, cannot handle string-based images."
            )
        loaded_img = Image.open(img)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        loaded_img.save(full_path)
    else:
        raise ValueError(
            "Cannot save image. It must be a PIL.Image.Image or "
            "a file path to an existing image."
        )
        
@ensure_pkg(
    "PIL", 
    extra="'PIL' is required for 'resize_image' to proceed."
)
def resize_image(
    image_path: str,
    size: Tuple[int, int],
    save_path: Optional[str] = None,
    maintain_aspect_ratio: bool = True,
    resample: Optional[int] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Resize an image from a file path to a new size, optionally
    maintaining the aspect ratio. Can also save the resized
    result if a `save_path` is provided.

    Parameters
    ----------
    image_path : str
        The file path or identifier to the source image. This
        path is validated or loaded through :func:`check_image`.
    size : tuple of (int, int)
        The new size (width, height) for the image. If
        ``maintain_aspect_ratio=True``, it only uses this as a
        bounding box, preserving the original proportions.
    save_path : str, optional
        Where to save the resized image. If not provided, the
        function only returns the resized PIL image without
        saving. If provided, the directory is created if missing.
    maintain_aspect_ratio : bool, optional
        If ``True``, uses :func:`PIL.Image.thumbnail` to
        preserve aspect ratio. Otherwise, uses
        :func:`PIL.Image.resize`.
    resample : int, optional
        Resampling filter, e.g. ``Image.BILINEAR``, ``Image.BICUBIC``,
        or ``Image.ANTIALIAS``. Defaults to ``Image.ANTIALIAS``
        if PIL is installed. If PIL is missing, it uses a fallback.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs printed.
        * 1 : Basic status messages.
        * 2+ : More detailed logs (e.g., original vs.
          new size).

    Returns
    -------
    img : PIL.Image.Image
        The resized image as a PIL image object. If you
        want to persist this on disk, a `save_path` must be
        provided or use :func:`save_image` separately.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist (and `error='raise'` in
        underlying checks).
    ValueError
        If the `size` is not a tuple of exactly two integers, or
        if ``maintain_aspect_ratio=True`` is used with invalid
        bounding sizes.
    ImportError
        If PIL is not installed, can't proceed with image
        manipulations.

    Notes
    -----
    This function uses :func:`check_image` under the hood to
    ensure the file is a valid image. For saving, it calls
    :func:`save_image`, making the code more robust and
    consistent with the rest of the library [1]_.

    Examples
    --------
    >>> from gofast.utils.image import resize_image
    >>> resized_img = resize_image(
    ...     "path/to/image.jpg",
    ...     size=(800, 600),
    ...     save_path="output/resized_image.jpg",
    ...     maintain_aspect_ratio=False,
    ...     verbose=2
    ... )
    Resized image from (1920, 1080) to (800, 600)
    without maintaining aspect ratio.
    Resized image saved to 'output/resized_image.jpg'.

    See Also
    --------
    check_image : Validate an image is loadable or meets
        certain format criteria.
    save_image : Write a PIL image to disk with flexible
        naming logic.

    References
    ----------
    .. [1] Clark, J., Freedman, D., & Hopwood, C. (2020).
           *Image-based transformations in Python with PIL*.
           *Python Imaging Library Official Documentation*.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot resize images."
        )
    resample = resample or (Image.ANTIALIAS if Image else 1)
    
    # Step 1: Validate the image path or object
    # If this fails, it raises an exception or returns None
    # based on 'ops' logic in check_image. We always want
    # to load the actual image object => ops='validate'.
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")

    pil_img = check_image(
        img=image_path,
        ops='validate',      # Return a PIL image instance if valid
        formats=None         # Or pass a list if format check is needed
    )
    if not pil_img:
        # Something went wrong in check_image
        raise ValueError(
            f"Could not validate the image: {image_path}."
        )

    # Step 2: Ensure 'size' is valid: (width, height) integers
    size = validate_length_range(
        size, sorted_values=False, 
        param_name="Size"
    )
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(isinstance(dim, int) for dim in size)
    ):
        raise ValueError(
            "Size must be a tuple of two integers (width, height)."
        )

    original_size = pil_img.size
    # Step 3: Resize logic
    if maintain_aspect_ratio:
        # Attempt bounding with thumbnail
        pil_img.thumbnail(size, resample)
        resized_size = pil_img.size
        if verbose >= 1:
            print(
                f"Resized image from {original_size} "
                f"to {resized_size} while maintaining "
                "aspect ratio."
            )
    else:
        # Hard resize ignoring ratio
        pil_img = pil_img.resize(size, resample)
        resized_size = pil_img.size
        if verbose >= 1:
            print(
                f"Resized image from {original_size} "
                f"to {resized_size} without maintaining "
                "aspect ratio."
            )

    # Step 4: Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving resized image to: {save_path}")
        # Use our save_image utility
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Resized image saved to '{save_path}'.")

    # Return a copy to ensure the loaded file is closed
    return pil_img.copy()

def crop_image(
    image_path: str,
    crop_box: Tuple[int, int, int, int],
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Crop an image to a specified bounding box and optionally
    save the result. Cropping is performed using the box
    specified by `(left, upper, right, lower)` coordinates.

    Parameters
    ----------
    image_path : str
        Path to the source image. Validated and loaded via
        :func:`check_image`.
    crop_box : tuple of (int, int, int, int)
        Bounding coordinates as ``(left, upper, right, lower)``.
        These must lie within the original image dimensions.
    save_path : str, optional
        If provided, the cropped image is saved at this path
        using :func:`save_image`. If no path is given, the
        function only returns the cropped image.
    verbose : int, optional
        Verbosity level:
        
        * 0 : No logs printed.
        * 1 : Basic info about bounding box and final action.
        * 2+ : More debug-level information.

    Returns
    -------
    cropped_img : PIL.Image.Image
        The newly cropped image as a PIL object. If you want
        the data persisted, use `save_path` or call
        :func:`save_image` afterwards.

    Raises
    ------
    FileNotFoundError
        If ``image_path`` is not found (via :func:`check_image`).
    ValueError
        If ``crop_box`` is invalid or outside the image
        boundaries. Also raised if the bounding box does
        not contain strictly increasing coordinates
        `(left < right, upper < lower)`.
    ImportError
        If PIL is not installed and we attempt image operations.

    Notes
    -----
    This function calls :func:`check_image` with
    ``ops='validate'`` to ensure a loadable image. The
    ``crop_box`` must satisfy:

    .. math::
       0 \le \text{left} < \text{right} \le \text{image\_width},
       \quad
       0 \le \text{upper} < \text{lower} \le \text{image\_height}.

    Examples
    --------
    >>> from gofast.utils.image import crop_image
    >>> cropped = crop_image(
    ...     "path/to/image.jpg",
    ...     (100, 50, 400, 300),
    ...     save_path="output/cropped_image.jpg",
    ...     verbose=1
    ... )
    Cropped image from box (100, 50, 400, 300).
    Cropped image saved to 'output/cropped_image.jpg'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot crop images."
        )

    # 1) Validate and load the image
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate',
        formats=None
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load the image for cropping: {image_path}"
        )

    # 2) Validate `crop_box`
    if (
        not isinstance(crop_box, tuple)
        or len(crop_box) != 4
        or not all(isinstance(coord, int) for coord in crop_box)
    ):
        raise ValueError(
            "crop_box must be a tuple of four integers "
            "(left, upper, right, lower)."
        )

    img_width, img_height = pil_img.size
    left, upper, right, lower = crop_box

    # Ensure coordinates lie within the image and are strictly increasing
    if not (0 <= left < right <= img_width) or not (0 <= upper < lower <= img_height):
        raise ValueError(
            f"crop_box={crop_box} is out of image bounds or invalid."
        )

    # 3) Crop the image
    cropped_img = pil_img.crop(crop_box)
    if verbose >= 1:
        print(f"Cropped image from box {crop_box} on size {pil_img.size}.")

    # 4) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving cropped image to: {save_path}")
        save_image(
            img=cropped_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Cropped image saved to '{save_path}'.")

    return cropped_img.copy()

@ensure_pkg(
    "PIL", 
    extra="'PIL' (Pillow) is required for 'rotate_image' to proceed."
)
def rotate_image(
    image_path: str,
    angle: float,
    save_path: Optional[str] = None,
    expand: bool = True,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Rotate an image by a specified angle (in degrees) and
    optionally save the result. By default, it expands the
    output to accommodate the entire rotated image.

    Parameters
    ----------
    image_path : str
        Path to the image to rotate. Validated via
        :func:`check_image`.
    angle : float
        Rotation angle in degrees. Positive values rotate
        counterclockwise.
    save_path : str, optional
        If provided, the rotated image is saved at this path
        using :func:`save_image`. If not, only returns the
        rotated image object.
    expand : bool, optional
        If ``True``, expands the output image to avoid cropping
        corners. If ``False``, keeps the original size with
        potential trimming of edges.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs printed.
        * 1 : Basic info messages (angle, success logs).
        * 2+ : More debug-level data.

    Returns
    -------
    rotated_img : PIL.Image.Image
        The rotated image as a PIL object.

    Raises
    ------
    FileNotFoundError
        If the file is not found or invalid based on
        :func:`check_image` calls.
    ImportError
        If PIL is not available.

    Examples
    --------
    >>> from gofast.utils.image import rotate_image
    >>> rotated = rotate_image(
    ...     "path/to/image.png",
    ...     45.0,
    ...     save_path="output/rotated_image.png",
    ...     verbose=1
    ... )
    Rotated image by 45.0 degrees.
    Rotated image saved to 'output/rotated_image.png'.

    Notes
    -----
    This function uses :func:`check_image` with
    ``ops='validate'`` to load the image. The rotation is done
    via :meth:`PIL.Image.Image.rotate`.

    See Also
    --------
    crop_image : Cropping images to a bounding box.
    resize_image : Resizing images to new dimensions.
    PIL.Image.rotate : The underlying Pillow method for
        rotation.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot rotate images."
        )

    # 1) Validate and load the image
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate',
        formats=None
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load the image for rotation: {image_path}"
        )

    # 2) Rotate
    rotated_img = pil_img.rotate(angle, expand=expand)
    if verbose >= 1:
        print(f"Rotated image by {angle} degrees.")

    # 3) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving rotated image to: {save_path}")
        save_image(
            img=rotated_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Rotated image saved to '{save_path}'.")

    return rotated_img.copy()

def flip_image(
    image_path: str,
    direction: str = 'horizontal',
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Flip an image either horizontally or vertically. The
    flipped result can be saved to disk if a `save_path`
    is provided.

    Parameters
    ----------
    image_path : str
        Path to the image file to flip. Validated and
        loaded using :func:`check_image`.
    direction : {'horizontal', 'vertical'}, optional
        Direction of the flip. If ``'horizontal'``, the
        image is mirrored left-right. If ``'vertical'``,
        the image is mirrored top-bottom.
    save_path : str, optional
        If provided, the flipped image is saved to this
        path using :func:`save_image`. Otherwise, only
        returns the flipped PIL image.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs printed.
        * 1 : Basic info (flip direction, final saving).
        * 2+ : Potentially more debug logs.

    Returns
    -------
    flipped_img : PIL.Image.Image
        A copy of the flipped image as a PIL object.

    Raises
    ------
    ValueError
        If the `direction` is not one of ``{'horizontal',
        'vertical'}``.
    FileNotFoundError
        If the given `image_path` is not found or invalid
        per :func:`check_image`.
    ImportError
        If PIL is not installed.

    Notes
    -----
    Internally uses :meth:`PIL.Image.Image.transpose` with
    ``Image.FLIP_LEFT_RIGHT`` or ``Image.FLIP_TOP_BOTTOM``.

    Examples
    --------
    >>> from gofast.utils.image import flip_image
    >>> flipped = flip_image(
    ...     "input/image.png",
    ...     direction="vertical",
    ...     save_path="output/flipped_vertical.png",
    ...     verbose=1
    ... )
    Flipped image vertically.
    Flipped image saved to 'output/flipped_vertical.png'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot flip images."
        )

    # 1) Validate and load image
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(f"Could not load the image for flipping: {image_path}")

    # 2) Determine flip direction
    if direction == 'horizontal':
        flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        if verbose >= 1:
            print("Flipped image horizontally.")
    elif direction == 'vertical':
        flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        if verbose >= 1:
            print("Flipped image vertically.")
    else:
        raise ValueError(
            "Direction must be 'horizontal' or 'vertical'. "
            f"Got {direction}."
        )

    # 3) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving flipped image to: {save_path}")
        save_image(
            img=flipped,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Flipped image saved to '{save_path}'.")

    return flipped.copy()

def convert_image_format(
    image_path: str,
    target_format: str,
    save_path: Optional[str] = None,
    verbose: int = 0
) -> str:
    r"""
    Convert an image file to a different format (e.g. PNG
    to JPG). Loads the image using :func:`check_image`,
    then re-saves in the desired format.

    Parameters
    ----------
    image_path : str
        Path to the original image file for conversion.
    target_format : str
        Desired format (e.g. 'png', 'jpeg', 'webp').
    save_path : str, optional
        If provided, uses that as the complete output path.
        Otherwise, creates a new file path by replacing
        the existing extension with `.target_format`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs printed.
        * 1 : Basic info about conversion and final name.
        * 2+ : More debug logs if needed.

    Returns
    -------
    new_file_path : str
        The file path where the new format image is saved.

    Raises
    ------
    FileNotFoundError
        If the input file is not found or invalid.
    ImportError
        If PIL is unavailable.
    ValueError
        If `target_format` is invalid or unknown to PIL.

    Examples
    --------
    >>> from gofast.utils.image import convert_image_format
    >>> out_path = convert_image_format(
    ...     "input/image.jpg",
    ...     "png",
    ...     save_path="output/converted_image.png",
    ...     verbose=1
    ... )
    Converted image to PNG format.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot convert images."
        )

    # 1) Validate and load the image
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(f"Could not load image for conversion: {image_path}")

    # 2) Determine save_path
    if not save_path:
        # Replace extension
        if '.' in image_path:
            base_path = image_path.rsplit('.', 1)[0]
        else:
            base_path = image_path  # no extension
        save_path = f"{base_path}.{target_format.lower()}"

    # 3) Convert and save
    if verbose >= 1:
        print(f"Converting image to {target_format.upper()} format.")
    # Use the PIL save method with format
    # E.g. target_format='PNG' or 'JPEG'
    # We'll standardize to upper
    pil_img.save(save_path, target_format.upper())

    if verbose >= 1:
        print(f"Image saved to '{save_path}'.")

    return save_path

def adjust_image(
    image_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Adjust the brightness, contrast, and saturation of
    an image by specified factors, optionally saving
    the result.

    Parameters
    ----------
    image_path : str
        Path to the input image. Validated and opened
        via :func:`check_image`.
    brightness : float, optional
        Factor for brightness enhancement. A value of 1.0
        means no change, <1.0 darkens, >1.0 brightens.
    contrast : float, optional
        Factor for contrast. 1.0 means no change,
        <1.0 lowers contrast, >1.0 increases contrast.
    saturation : float, optional
        Factor for saturation. 1.0 means no change,
        <1.0 de-saturates, >1.0 intensifies color.
    save_path : str, optional
        If provided, the adjusted image is saved to this
        path. Otherwise, it is only returned.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs.
        * 1 : Basic info on changes applied.
        * 2+ : More details if needed.

    Returns
    -------
    adjusted_img : PIL.Image.Image
        The image with adjustments applied.

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid or absent.
    ImportError
        If PIL is not installed.
    ValueError
        If any of the enhancement factors are invalid.

    Notes
    -----
    This function uses the :class:`PIL.ImageEnhance.Brightness`,
    :class:`PIL.ImageEnhance.Contrast`, and
    :class:`PIL.ImageEnhance.Color` classes internally.

    Examples
    --------
    >>> from gofast.utils.image import adjust_image
    >>> out_img = adjust_image(
    ...     "input/photo.jpg",
    ...     brightness=1.2,
    ...     contrast=0.8,
    ...     saturation=1.5,
    ...     save_path="output/adjusted_photo.jpg",
    ...     verbose=1
    ... )
    Applied brightness=1.2, contrast=0.8, saturation=1.5.
    Image saved to 'output/adjusted_photo.jpg'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot adjust images."
        )

    # 1) Validate and load
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(f"Could not load image for adjustments: {image_path}")

    # 2) Apply brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)

    # 3) Apply contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)

    # 4) Apply saturation
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(saturation)

    if verbose >= 1:
        print(
            f"Applied brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}."
        )

    # 5) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving adjusted image to: {save_path}")
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Image saved to '{save_path}'.")

    return pil_img.copy()


def add_watermark(
    image_path: str,
    watermark: str,
    position: Tuple[int, int] = (0, 0),
    save_path: Optional[str] = None,
    opacity: int = 128,
    scale: float = 0.3,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Add a watermark onto an image at a specified position
    with optional opacity and scaling. By default, it resizes
    the watermark to 30% of the base image dimensions.

    Parameters
    ----------
    image_path : str
        The path to the base image. Validated and loaded
        via :func:`check_image`.
    watermark : str
        The path to the watermark image (e.g. PNG).
    position : tuple of (int, int), optional
        Top-left coordinates where the watermark is placed.
        Default is (0, 0).
    save_path : str, optional
        If provided, the result is saved to this path using
        :func:`save_image`. Otherwise, only the watermarked
        image object is returned.
    opacity : int, optional
        The alpha value (0-255) for watermark transparency.
        Lower => more transparent. Default=128 (semi-transparent).
    scale : float, optional
        Fractional scale of the watermark relative to the
        base image size. A value of 0.3 means 30% of the
        base dimension.
    verbose : int, optional
        Verbosity level:
        
        * 0 : No logs.
        * 1 : Basic info about watermark sizing, final saving.
        * 2+ : More debug info.

    Returns
    -------
    watermarked_img : PIL.Image.Image
        A copy of the resultant watermarked image in RGBA
        mode.

    Raises
    ------
    FileNotFoundError
        If either ``image_path`` or ``watermark`` is
        invalid or does not exist.
    ImportError
        If Pillow (PIL) is not installed.
    ValueError
        If scale <= 0, or opacity not in [0,255], etc.

    Notes
    -----
    This function ensures both images are in RGBA mode so
    that alpha compositing can be applied. The watermark
    is scaled, its alpha is adjusted, then pasted over
    the base at the given position.

    Examples
    --------
    >>> from gofast.utils.image import add_watermark
    >>> result_img = add_watermark(
    ...     "input/photo.jpg",
    ...     "watermarks/logo.png",
    ...     position=(50, 50),
    ...     save_path="output/photo_watermarked.png",
    ...     opacity=180,
    ...     scale=0.2,
    ...     verbose=1
    ... )
    Loaded base image and watermark. Watermark scaled to
    (200 x 150), alpha=180. Watermarked image saved to
    'output/photo_watermarked.png'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) not installed. Cannot apply watermarks."
        )

    if scale <= 0:
        raise ValueError(
            f"Scale must be positive. Got {scale}"
        )
    if not (0 <= opacity <= 255):
        raise ValueError(
            f"Opacity must be between 0 and 255. Got {opacity}"
        )

    # 1) Validate and load base image
    if verbose >= 1:
        print(f"Validating base image: {image_path}")
    base = check_image(
        img=image_path,
        ops='validate'
    )
    if base is None:
        raise ValueError(
            f"Failed to load base image: {image_path}"
        )

    # 2) Validate and load watermark
    if verbose >= 1:
        print(f"Validating watermark image: {watermark}")
    wm_img = check_image(
        img=watermark,
        ops='validate'
    )
    if wm_img is None:
        raise ValueError(
            f"Failed to load watermark: {watermark}"
        )

    # Convert to RGBA
    base = base.convert("RGBA")
    wm_img = wm_img.convert("RGBA")

    # 3) Scale watermark
    new_wm_width = int(base.width * scale)
    new_wm_height = int(base.height * scale)
    if verbose >= 2:
        print(
            f"Resizing watermark to {new_wm_width}x{new_wm_height} "
            f"(scale={scale})."
        )
    wm_img = wm_img.resize((new_wm_width, new_wm_height))

    # 4) Adjust watermark opacity
    wm_img.putalpha(opacity)
    if verbose >= 1:
        print(
            f"Watermark alpha set to {opacity}."
        )

    # 5) Paste watermark onto base
    base.paste(wm_img, position, wm_img)
    if verbose >= 1:
        print(
            f"Watermark placed at {position} "
            f"on image size {base.size}."
        )

    # 6) Optionally save
    if save_path:
        if verbose >= 2:
            print(
                f"Saving watermarked image to: {save_path}"
            )
        save_image(
            img=base,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(
                f"Watermarked image saved to '{save_path}'."
            )

    return base.copy()

def compress_image(
    image_path: str,
    quality: int = 85,
    save_path: Optional[str] = None,
    optimize: bool = True,
    verbose: int = 0
) -> str:
    r"""
    Compress (and optionally re-save) an image with a
    given JPEG quality setting. Typically used for JPEG
    compression but can also apply to PNG, etc.  

    Parameters
    ----------
    image_path : str
        Path to the input image.
    quality : int, optional
        Quality setting (1 to 95+). Higher values
        => higher quality & bigger size. Defaults to 85.
    save_path : str, optional
        Where to store the compressed image. If not
        provided, overwrites the original file. 
    optimize : bool, optional
        Whether to optimize the Huffman tables
        (lossless step). Improves file size slightly
        without quality loss. Default: True.
    verbose : int, optional
        Verbosity:

        * 0 : No logs.
        * 1 : Logs about final action.
        * 2+ : More debug.

    Returns
    -------
    compressed_path : str
        The path where the compressed file is saved.

    Raises
    ------
    FileNotFoundError
        If the source file is invalid per checks.
    ImportError
        If PIL is not installed.
    ValueError
        If quality is out of typical range.

    Notes
    -----
    This function is primarily for JPEG, but can also
    apply to other formats. Keep in mind certain formats
    (like PNG) might not respect "quality" in the same
    way as JPEG.

    Examples
    --------
    >>> from gofast.utils.image import compress_image
    >>> out_path = compress_image(
    ...     "input/photo.jpg",
    ...     quality=70,
    ...     save_path="output/photo_compressed.jpg",
    ...     verbose=1
    ... )
    Compressed image saved to 'output/photo_compressed.jpg'
    with quality=70.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot compress images."
        )

    if not (1 <= quality <= 100):
        raise ValueError(
            "Quality must be between 1 and 100. "
            f"Got {quality}."
        )

    # 1) Validate & load the image
    if verbose >= 1:
        print(f"Validating image for compression: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load image for compression: {image_path}"
        )

    # 2) Determine final path
    if not save_path:
        # Overwrite the original by default
        save_path = image_path

    # 3) Use PIL save with compression settings
    # For typical JPEG usage:
    pil_img.save(
        save_path,
        optimize=optimize,
        quality=quality
    )

    if verbose >= 1:
        print(
            f"Compressed image saved to '{save_path}' "
            f"with quality={quality}."
        )

    return save_path


def apply_filter(
    image_path: str,
    filter_type: str = 'grayscale',
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Apply a simple filter (grayscale, sepia, etc.) to an
    image and optionally save it.

    Parameters
    ----------
    image_path : str
        Path to the source image. Validated via
        :func:`check_image`.
    filter_type : {'grayscale', 'sepia', ...}, optional
        The type of filter to apply. Currently supports:
        
        * ``'grayscale'``: Convert to 8-bit grayscale.
        * ``'sepia'``: Sepia-toned transformation.
        
        More filters can be added or extended as needed.
    save_path : str, optional
        If provided, saves the filtered image to this path
        via :func:`save_image`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs
        * 1 : Basic info about the filter
        * 2+ : Potential debug details

    Returns
    -------
    filtered_img : PIL.Image.Image
        The newly filtered image object. If you wish to
        persist, specify `save_path` or call
        :func:`save_image`.

    Raises
    ------
    FileNotFoundError
        If the file is invalid per checks.
    ImportError
        If PIL is unavailable.
    ValueError
        If `filter_type` is unsupported.

    Notes
    -----
    * The 'sepia' filter uses a simple transform
      matrix on the RGB channels, clamping to 255.
    * Additional filters can be integrated by
      modifying this function or referencing other
      PIL transformations.

    Examples
    --------
    >>> from gofast.utils.image import apply_filter
    >>> filtered = apply_filter(
    ...     "input/photo.png",
    ...     filter_type="sepia",
    ...     save_path="output/photo_sepia.png",
    ...     verbose=1
    ... )
    Applied filter: sepia
    Filtered image saved to 'output/photo_sepia.png'
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. "
            "Cannot apply image filters."
        )

    # 1) Validate & load
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load image to apply filter: {image_path}"
        )

    # 2) Apply filter
    if filter_type == 'grayscale':
        pil_img = pil_img.convert('L')
    elif filter_type == 'sepia':
        sepia = Image.new('RGB', pil_img.size)
        pixels = pil_img.convert('RGB').getdata()
        new_pixels = []
        for pixel in pixels:
            r, g, b = pixel
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            new_pixels.append(
                (min(tr, 255), min(tg, 255), min(tb, 255))
            )
        sepia.putdata(new_pixels)
        pil_img = sepia
    else:
        raise ValueError(
            f"Unsupported filter_type: {filter_type}."
        )

    if verbose >= 1:
        print(f"Applied filter: {filter_type}")

    # 3) Optionally save
    if save_path:
        if verbose >= 2:
            print(
                f"Saving filtered image to: {save_path}"
            )
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(
                f"Filtered image saved to '{save_path}'"
            )

    return pil_img.copy()

def generate_thumbnail(
    image_path: str,
    size: Tuple[int, int] = (128, 128),
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Generate a thumbnail for an image at the given size,
    preserving aspect ratio. Useful for quickly creating
    previews or icons.

    Parameters
    ----------
    image_path : str
        Path to the source image file. Validated using
        :func:`check_image`.
    size : (int, int), optional
        Desired maximum size (width, height). The image is
        resized so that neither dimension exceeds these
        values, preserving its aspect ratio.
    save_path : str, optional
        If provided, saves the thumbnail to this path with
        :func:`save_image`. If not, only returns the
        thumbnail image.
    verbose : int, optional
        Verbosity level:
        
        * 0 : No logs.
        * 1 : Basic info (size, final path).
        * 2+ : More debug.

    Returns
    -------
    thumbnail_img : PIL.Image.Image
        The thumbnail as a PIL Image object. If you want
        it persisted, supply `save_path`.

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid or absent.
    ImportError
        If PIL is unavailable.
    ValueError
        If size is invalid or negative.

    Notes
    -----
    Internally calls :meth:`PIL.Image.Image.thumbnail`,
    which modifies the image in place to fit within
    the bounding box.

    Examples
    --------
    >>> from gofast.utils.image import generate_thumbnail
    >>> thumb = generate_thumbnail(
    ...     "input/photo.jpg",
    ...     size=(200, 200),
    ...     save_path="output/photo_thumb.jpg",
    ...     verbose=1
    ... )
    Generated thumbnail with max size=(200, 200).
    Thumbnail saved to 'output/photo_thumb.jpg'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot create thumbnails."
        )

    # 1) Validate & load image
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load image for thumbnail: {image_path}"
        )

    # 2) Validate size
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in size)
    ):
        raise ValueError(
            "Size must be a (width, height) tuple of positive integers."
        )

    # 3) Generate thumbnail
    pil_img.thumbnail(size)
    if verbose >= 1:
        print(f"Generated thumbnail with max size={size}.")

    # 4) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving thumbnail to: {save_path}")
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Thumbnail saved to '{save_path}'.")

    return pil_img.copy()


def extract_metadata(
    image_path: str,
    verbose: int = 0
) -> dict:
    r"""
    Extract metadata (EXIF info) from an image, if available.

    Parameters
    ----------
    image_path : str
        Path to the image file. Validated & loaded via
        :func:`check_image`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs
        * 1 : Logs success
        * 2+ : Potential debug

    Returns
    -------
    metadata : dict
        A dictionary of EXIF tags and their values if
        present, otherwise an empty dict.

    Raises
    ------
    FileNotFoundError
        If the image is not found.
    ImportError
        If PIL is unavailable.

    Notes
    -----
    Some images (e.g., PNG) may not have EXIF data. In such
    cases, this returns an empty dictionary [1]_.

    Examples
    --------
    >>> from gofast.utils.image import extract_metadata
    >>> info = extract_metadata(
    ...     "input/photo.jpg",
    ...     verbose=1
    ... )
    Found EXIF data with 10 tags. # Example
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot extract metadata."
        )

    # 1) Validate & load
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load image for metadata extraction: {image_path}"
        )

    # 2) Extract EXIF
    # PIL's _getexif() returns dict or None
    meta = pil_img._getexif() or {}
    if verbose >= 1:
        print(
            f"Extracted {len(meta)} metadata tags from the image."
        )
    return meta


def overlay_text(
    image_path: str,
    text: str,
    position: Tuple[int, int] = (10, 10),
    font_path: Optional[str] = None,
    font_size: int = 20,
    color: Tuple[int, int, int] = (255, 255, 255),
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Overlay text onto an image at a given position, using
    a custom or default font, then optionally save the
    result.

    Parameters
    ----------
    image_path : str
        Path to the source image, validated using
        :func:`check_image`.
    text : str
        The text to place on the image.
    position : (int, int), optional
        Coordinates (x, y) from the top-left corner for
        the text.
    font_path : str, optional
        Path to a TrueType or OpenType font file. If not
        provided, uses PIL's default font.
    font_size : int, optional
        Size of the text in points. Default=20.
    color : (int, int, int), optional
        The RGB color for the text. Default=white.
    save_path : str, optional
        If provided, the modified image is saved to this
        path via :func:`save_image`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs
        * 1 : Basic info about text overlay
        * 2+ : Additional debug info.

    Returns
    -------
    overlaid_img : PIL.Image.Image
        The image with text drawn onto it.

    Raises
    ------
    FileNotFoundError
        If the image file is not found or invalid.
    ImportError
        If Pillow is unavailable.
    OSError
        If the specified font_path cannot be loaded.
    ValueError
        If the position or size is invalid.

    Notes
    -----
    This function ensures the image is in RGBA mode if
    necessary for compositing. The text is rendered via
    :class:`PIL.ImageDraw.Draw`.

    Examples
    --------
    >>> from gofast.utils.image import overlay_text
    >>> out = overlay_text(
    ...     "input/photo.png",
    ...     "Hello World",
    ...     position=(50, 100),
    ...     font_path="fonts/Arial.ttf",
    ...     font_size=24,
    ...     color=(255, 0, 0),
    ...     save_path="output/photo_text.png",
    ...     verbose=1
    ... )
    Drew text "Hello World" at position (50, 100).
    Overlaid image saved to 'output/photo_text.png'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot overlay text on images."
        )

    # 1) Validate & load
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load the image to overlay text: {image_path}"
        )

    # Convert to RGBA for consistent text overlay
    pil_img = pil_img.convert("RGBA")

    # 2) Prepare drawing context
    if verbose >= 2:
        print(
            f"Creating draw context with font_path={font_path}, "
            f"font_size={font_size}"
        )
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_img)
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except OSError as e:
        raise OSError(
            f"Could not load font from path: {font_path}"
        ) from e

    # 3) Draw text
    draw.text(position, text, font=font, fill=color)
    if verbose >= 1:
        print(
            f'Drew text "{text}" at position {position}.'
        )

    # 4) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving overlaid image to: {save_path}")
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Overlaid image saved to '{save_path}'.")

    return pil_img.copy()

@validate_params ({
    "target_space": [StrOptions( {'HSV', 'GRAY', 'gray', 'hsv'})]
  })
def convert_color_space(
    image_path: str,
    target_space: str = 'HSV',
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "np.ndarray":
    r"""
    Convert an image's color space using OpenCV (e.g., BGR
    to HSV or GRAY). This function is not PIL-based, so it
    calls :func:`check_files` for path validation. The
    result is returned as a NumPy array.

    Parameters
    ----------
    image_path : str
        Path to the image file. Validated via
        :func:`check_files`.
    target_space : {'HSV', 'GRAY'}, optional
        Desired color space. Currently supports:
        
        * ``'HSV'``: Converts from BGR to HSV.
        * ``'GRAY'``: Converts from BGR to Grayscale.
    save_path : str, optional
        If provided, saves the converted array using
        OpenCV's ``cv2.imwrite``.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs
        * 1 : Basic info about color space and final path
        * 2+ : More details if needed

    Returns
    -------
    converted : numpy.ndarray
        The image data in the new color space, as an
        OpenCV-compatible array.

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid as per :func:`check_files`.
    ImportError
        If OpenCV (`cv2`) is not installed.
    ValueError
        If `target_space` is unsupported.

    Notes
    -----
    The input is assumed to be in BGR format, as typically
    read by OpenCV (``cv2.imread``). The function uses
    ``cv2.cvtColor`` with the appropriate flags:

    .. math::
        \text{BGR} \rightarrow \text{HSV} \quad
        \text{BGR} \rightarrow \text{GRAY}

    Examples
    --------
    >>> from gofast.utils.image import convert_color_space
    >>> hsv_data = convert_color_space(
    ...     "input/img.jpg",
    ...     target_space='HSV',
    ...     save_path="output/img_hsv.png",
    ...     verbose=1
    ... )
    Converted image to HSV space.
    Saved to 'output/img_hsv.png'.
    """
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is not installed. Cannot convert color space."
        )

    # 1) Validate file existence
    if verbose >= 1:
        print(f"Validating file for color space conversion: {image_path}")
    checked = check_files(
        files=image_path,
        formats=None,
        return_valid=True,  # we only need to see if it's valid
        error='raise',
        empty_allowed=False
    )
    if not checked:
        raise ValueError(
            f"File is invalid for color conversion: {image_path}"
        )

    # 2) Load via OpenCV
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise ValueError(
            f"OpenCV could not load the image: {image_path}"
        )

    # 3) Convert
    target_space_up = target_space.upper()
    if target_space_up == 'HSV':
        converted = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    elif target_space_up == 'GRAY':
        converted = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Unsupported color space: {target_space}. "
            "Choose 'HSV' or 'GRAY'."
        )

    if verbose >= 1:
        print(f"Converted image to {target_space_up} space.")

    # 4) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Writing converted file to: {save_path}")
        cv2.imwrite(save_path, converted)
        if verbose >= 1:
            print(f"Saved to '{save_path}'.")

    return converted


def detect_faces(
    image_path: str,
    cascade_path: str = 'haarcascade_frontalface_default.xml',
    save_path: Optional[str] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    verbose: int = 0
) -> Tuple[List[Tuple[int, int, int, int]], "np.ndarray"]:
    r"""
    Detect faces in an image using OpenCV's Haar Cascade
    detection. Optionally, draw bounding boxes around
    detected faces and save the result.

    Parameters
    ----------
    image_path : str
        Path to the source image. Validated with
        :func:`check_files`.
    cascade_path : str, optional
        The filename of the Haar Cascade (e.g.
        `'haarcascade_frontalface_default.xml'`).
        If not an absolute path, it uses
        ``cv2.data.haarcascades + cascade_path`` by default.
    save_path : str, optional
        If provided, writes the annotated image (with
        bounding boxes) to this path via ``cv2.imwrite``.
    scale_factor : float, optional
        Parameter specifying how much the image size is
        reduced at each image scale. Default=1.1.
    min_neighbors : int, optional
        Parameter specifying how many neighbors each
        candidate rectangle should have to retain it.
        Default=4.
    verbose : int, optional
        Verbosity level:

        * 0 : No console logs.
        * 1 : Basic info on detections and saving.
        * 2+ : More debug-level logs.

    Returns
    -------
    faces : list of tuples
        A list of bounding boxes for detected faces. Each
        box is (x, y, w, h).
    annotated_img : numpy.ndarray
        The loaded image array in BGR format with
        bounding boxes drawn on it.

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid or the cascade path
        is not found by OpenCV.
    ImportError
        If OpenCV is not installed.
    ValueError
        If the image could not be loaded properly
        via OpenCV.

    Notes
    -----
    This function uses :func:`check_files` to ensure the
    file exists and is not empty. Then uses OpenCV to load
    and detect faces. The resulting bounding boxes are
    drawn if any are found, and optionally saved to
    `save_path`.

    Examples
    --------
    >>> from gofast.utils.image import detect_faces
    >>> faces, annotated = detect_faces(
    ...     "people.jpg",
    ...     save_path="people_annotated.jpg",
    ...     verbose=1
    ... )
    Detected 3 face(s).
    Annotated image saved to 'people_annotated.jpg'.
    """
    if cv2 is None:
        raise ImportError(
            "OpenCV is not installed. Cannot detect faces."
        )

    # 1) Validate image path
    if verbose >= 1:
        print(f"Validating file: {image_path}")
    valid_file = check_files(
        files=image_path,
        formats=None,
        return_valid=True,
        error='raise',
        empty_allowed=False
    )
    if not valid_file:
        raise ValueError(
            f"Invalid file or file not found: {image_path}"
        )

    # 2) Initialize Haar Cascade
    cascade_abs_path = (
        cascade_path
        if os.path.isabs(cascade_path)
        else os.path.join(cv2.data.haarcascades, cascade_path)
    )
    if verbose >= 2:
        print(f"Using cascade file: {cascade_abs_path}")
    face_cascade = cv2.CascadeClassifier(cascade_abs_path)
    if face_cascade.empty():
        raise FileNotFoundError(
            f"Could not load Haar Cascade: {cascade_abs_path}"
        )

    # 3) Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"OpenCV could not load the image: {image_path}"
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4) Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    if verbose >= 1:
        print(f"Detected {len(faces)} face(s).")

    # 5) Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # 6) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving annotated image to: {save_path}")
        cv2.imwrite(save_path, img)
        if verbose >= 1:
            print(f"Annotated image saved to '{save_path}'.")

    return faces, img

def enhance_image(
    image_path: str,
    enhancement_type: str = 'sharpen',
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Apply a simple enhancement to an image using Pillow
    filters (e.g., sharpen, blur). Optionally, save the
    result.

    Parameters
    ----------
    image_path : str
        Path to the input image, validated via
        :func:`check_image`.
    enhancement_type : {'sharpen', 'blur'}, optional
        The type of enhancement to apply:

        * ``'sharpen'``: Sharpens details in the image.
        * ``'blur'``: Blurs the image.

    save_path : str, optional
        If provided, the enhanced image is saved with
        :func:`save_image`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs.
        * 1 : Basic info about the enhancement.
        * 2+ : More debug.

    Returns
    -------
    enhanced_img : PIL.Image.Image
        The enhanced image object.

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid or not found.
    ImportError
        If PIL is unavailable.
    ValueError
        If `enhancement_type` is unsupported.

    Examples
    --------
    >>> from gofast.utils.image import enhance_image
    >>> output_img = enhance_image(
    ...     "input/photo.jpg",
    ...     enhancement_type="blur",
    ...     save_path="output/blur_photo.jpg",
    ...     verbose=1
    ... )
    Applied blur enhancement.
    Enhanced image saved to 'output/blur_photo.jpg'.
    """
    if Image is None or ImageFilter is None:
        raise ImportError(
            "PIL (Pillow) is not available. Cannot enhance images."
        )

    # 1) Validate & load
    if verbose >= 1:
        print(f"Validating image for enhancement: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load image for enhancement: {image_path}"
        )

    # 2) Apply chosen filter
    if enhancement_type == 'sharpen':
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
    elif enhancement_type == 'blur':
        pil_img = pil_img.filter(ImageFilter.BLUR)
    else:
        raise ValueError(
            f"Unsupported enhancement type: {enhancement_type}."
        )

    if verbose >= 1:
        print(f"Applied {enhancement_type} enhancement.")

    # 3) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving enhanced image to: {save_path}")
        save_image(
            img=pil_img,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Enhanced image saved to '{save_path}'.")

    return pil_img.copy()


def batch_process(
    directory: str,
    process_function,
    save_directory: Optional[str] = None,
    verbose: int = 0,
    **kwargs
) -> None:
    r"""
    Batch process all images in a directory using a user-defined
    function. Optionally save outputs in a separate directory.

    Parameters
    ----------
    directory : str
        The directory containing images. Each file is tested
        for typical image extensions. If a file is recognized,
        the `process_function` is called.
    process_function : callable
        A function that processes a single image. Must accept
        at least `img_path` and an optional `save_path`, plus
        other **kwargs. Typically one of the image utilities
        like `resize_image`, `enhance_image`, etc.
    save_directory : str, optional
        If provided, output files are saved there (with the
        same filename) or as the function's logic dictates.
        If not provided, the function might do in-place
        modifications or no saving at all, depending on
        `process_function` or **kwargs.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs
        * 1 : Basic info about how many files are processed
        * 2+ : More detailed logs (filenames, etc.)
    **kwargs : dict, optional
        Additional keyword arguments passed to
        `process_function`.

    Returns
    -------
    None
        This function doesn't return anything, but it calls
        the `process_function` on each recognized image.

    Raises
    ------
    FileNotFoundError
        If `directory` is not valid.
    ImportError
        If a required library for `process_function` is not
        installed.
    ValueError
        If `process_function` doesn't accept the required
        parameters or fails on some file.

    Notes
    -----
    This function only processes files whose extensions match
    a typical image pattern (`.png`, `.jpg`, `.jpeg`, `.bmp`,
    `.gif`). It ignores other files. The user-defined
    `process_function` should handle how to open/save in
    detail.

    Examples
    --------
    >>> from gofast.utils.image import batch_process, enhance_image
    >>> batch_process(
    ...     directory="input_images",
    ...     process_function=enhance_image,
    ...     save_directory="enhanced_images",
    ...     enhancement_type="sharpen",
    ...     verbose=1
    ... )
    Processed 10 images in 'input_images' -> 'enhanced_images'.
    """
    # 1) Check if directory is valid
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist."
        )

    # 2) Create save_directory if given and not exist
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
        if verbose >= 2:
            print(f"Created directory: {save_directory}")

    # 3) Collect valid image filenames
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    all_files = os.listdir(directory)
    image_files = [
        f for f in all_files
        if f.lower().endswith(valid_exts)
    ]
    if verbose >= 1:
        print(
            f"Found {len(image_files)} image(s) in directory '{directory}'."
        )

    # 4) Process each image
    count = 0
    for filename in image_files:
        img_path = os.path.join(directory, filename)
        save_path = (
            os.path.join(save_directory, filename)
            if save_directory
            else None
        )

        if verbose >= 2:
            print(f"Processing {img_path} -> {save_path}")

        try:
            process_function(
                image_path=img_path,
                save_path=save_path,
                **kwargs
            )
            count += 1
        except Exception as e:
            # User can handle or re-raise
            if verbose >= 2:
                print(f"Error processing {filename}: {e}")

    if verbose >= 1:
        print(
            f"Processed {count} image(s) in '{directory}'"
            f"{f' -> {save_directory}' if save_directory else ''}."
        )

def compare_images(
    image1_path: str,
    image2_path: str,
    save_path: Optional[str] = None,
    verbose: int = 0
) -> "Image.Image":
    r"""
    Compare two images pixel-by-pixel and produce a difference
    image. Pixels that are the same result in black, differing
    ones are highlighted, providing a quick way to see changes
    or differences.

    Parameters
    ----------
    image1_path : str
        Path to the first image. Validated via
        :func:`check_image`.
    image2_path : str
        Path to the second image. Also validated via
        :func:`check_image`.
    save_path : str, optional
        If provided, the resulting difference image is
        saved to this path using :func:`save_image`.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs.
        * 1 : Basic info about the images and final saving.
        * 2+ : More details if needed.

    Returns
    -------
    diff_img : PIL.Image.Image
        An image object highlighting differences. Typically
        black where images match, non-black where they differ.

    Raises
    ------
    FileNotFoundError
        If either image path is invalid per
        :func:`check_image`.
    ImportError
        If PIL is not installed.
    ValueError
        If the images cannot be loaded or differ in mode/size
        (which might cause an error with `ImageChops.difference`).

    Notes
    -----
    Under the hood, :func:`PIL.ImageChops.difference` is used,
    which computes the absolute pixel-wise difference between
    two images, ignoring alpha.

    Examples
    --------
    >>> from gofast.utils.image import compare_images
    >>> diff = compare_images(
    ...     "before.png", "after.png",
    ...     save_path="diff.png",
    ...     verbose=1
    ... )
    Compared images; difference saved to 'diff.png'.
    """
    if Image is None or ImageChops is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot compare images."
        )

    # 1) Validate & load image1
    if verbose >= 1:
        print(f"Checking and loading image1: {image1_path}")
    img1 = check_image(
        img=image1_path,
        ops='validate'
    )
    if img1 is None:
        raise ValueError(
            f"Could not load the first image: {image1_path}"
        )

    # 2) Validate & load image2
    if verbose >= 1:
        print(f"Checking and loading image2: {image2_path}")
    img2 = check_image(
        img=image2_path,
        ops='validate'
    )
    if img2 is None:
        raise ValueError(
            f"Could not load the second image: {image2_path}"
        )

    # 3) Compute difference
    diff = ImageChops.difference(img1, img2)
    if verbose >= 1:
        print(
            f"Compared images; found difference mode={diff.mode}, "
            f"size={diff.size}."
        )

    # 4) Optionally save
    if save_path:
        if verbose >= 2:
            print(f"Saving difference to: {save_path}")
        save_image(
            img=diff,
            save_path=save_path,
            save_anyway=True
        )
        if verbose >= 1:
            print(f"Difference saved to '{save_path}'.")

    return diff

def image_histogram(
    image_path: str,
    save_path: Optional[str] = None,
    verbose: int = 0
) -> list:
    r"""
    Generate a histogram of pixel intensity distribution
    for an image using Pillow's built-in histogram method.
    Optionally, produce a plot and save it.

    Parameters
    ----------
    image_path : str
        Path to the image. Validated via :func:`check_image`.
    save_path : str, optional
        If provided, a histogram plot is saved using
        matplotlib with 256 bins. The function does not
        display this plot, only saves it.
    verbose : int, optional
        Verbosity level:

        * 0 : No logs.
        * 1 : Basic info about the histogram, final saving.
        * 2+ : More details if needed.

    Returns
    -------
    hist_values : list of int
        A flat list of histogram counts. For an 8-bit color
        image in RGB, length is typically 768 (256 values
        for each channel: R, G, B).

    Raises
    ------
    FileNotFoundError
        If `image_path` is invalid or not found.
    ImportError
        If PIL is not installed.
    ValueError
        If the image cannot be loaded or is incompatible
        with hist operations.

    Notes
    -----
    For an 8-bit grayscale image, the histogram list has 256
    counts, each representing the number of pixels with that
    intensity. For RGB, each channel contributes 256 bins.

    Examples
    --------
    >>> from gofast.utils.image import image_histogram
    >>> hist = image_histogram(
    ...     "photo.jpg",
    ...     save_path="hist_plot.png",
    ...     verbose=1
    ... )
    Generated histogram with 768 values (RGB).
    Histogram plot saved to 'hist_plot.png'.
    """
    if Image is None:
        raise ImportError(
            "PIL (Pillow) is not installed. Cannot build histograms."
        )

    # 1) Validate & load
    if verbose >= 1:
        print(f"Checking and loading image: {image_path}")
    pil_img = check_image(
        img=image_path,
        ops='validate'
    )
    if pil_img is None:
        raise ValueError(
            f"Could not load the image for histogram: {image_path}"
        )

    # 2) Compute histogram
    hist = pil_img.histogram()
    if verbose >= 1:
        print(
            f"Generated histogram with {len(hist)} values."
        )

    # 3) Optionally plot & save
    if save_path:
        if verbose >= 2:
            print(f"Plotting histogram to: {save_path}")
        plt.figure()
        plt.hist(hist, bins=256, range=(0, 256), color='gray')
        plt.title("Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.savefig(save_path)
        plt.close()
        if verbose >= 1:
            print(f"Histogram plot saved to '{save_path}'.")

    return hist









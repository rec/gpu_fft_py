import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

cdef extern from "mailbox.h":
  int mbox_open()
  void mbox_close(int mb)
  const char* DEVICE_FILE_NAME

cdef extern from "gpu_fft.h":
  enum: GPU_FFT_QPUS

  cdef struct GPU_FFT_COMPLEX:
    float re, im

  cdef struct GPU_FFT_BASE:
    int mb
    unsigned handle, size, vc_msg, vc_code, peri_size
    unsigned vc_unifs[GPU_FFT_QPUS]
    unsigned* peri

  cdef struct GPU_FFT:
    GPU_FFT_BASE base
    GPU_FFT_COMPLEX* in_
    GPU_FFT_COMPLEX* out
    int x, y, step

  int gpu_fft_prepare(
      int mb,         # mailbox file_desc
      int log2_N,     # log2(FFT_length) = 8...17
      int direction,  # GPU_FFT_FWD: fft(); GPU_FFT_REV: ifft()
      int jobs,       # number of transforms in batch
      GPU_FFT **fft)

  unsigned gpu_fft_execute(GPU_FFT *info)
  void gpu_fft_release(GPU_FFT *info)


cdef class GpuFft:
  cdef GPU_FFT* fft
  cdef int mb, jobs, log_size, size
  cdef bool prepared
  cdef np.ndarray buffer

  def __cinit__(self, int log_size, is_forward=True, int jobs=10, buffer=None):
    cdef int forward, result
    if is_forward:
      forward = 1
    else:
      forward = 0

    self.jobs = jobs
    assert 8 <= log_size <= 17, 'log_size must be between 8 and 17'
    self.log_size = log_size
    self.size = 2 ** log_size
    self.prepared = False
    print('about to open device')
    self.mb = mbox_open()
    if self.mb < 0:
      print('Couldn\'t open device')
      raise Exception("Couldn't open device.")
    result = gpu_fft_prepare(self.mb, log_size, forward, jobs, &self.fft)
    if result < 0:
      if result == -1:
        err = 'Unable to enable V3D. Please check your firmware is up to date.'
      elif result == -2:
        err = 'log_size=%d not supported.  Try between 8 and 17.' % log_size
      elif result == -3:
        err = 'Out of memory.  Try a smaller batch or increase GPU memory.'
      else:
        err = 'Uknown error %d' % result
      raise Exception(err)
    self.prepared = True
    if buffer:
      self.buffer = buffer
    else:
      self.buffer = np.empty((self.jobs, self.size), dtype=complex)

  def __dealloc__(self):
    if self.prepared:
      gpu_fft_release(self.fft)
    if self.mb >= 0:
      mbox_close(self.mb)

  def execute(self, job_data):
    cdef float re, im
    cdef int i, j, data_count = 0, size_count = 0
    cdef GPU_FFT_COMPLEX* base
    for j in xrange(len(job_data)):
      data = job_data[j]
      assert j < self.jobs
      data_count = j
      base = self.fft.in_ + j * self.fft.step;
      for i in xrange(len(data)):
        d = data[i]
        assert i < self.size
        size_count = i
        try:
          re, im = d
        except:
          re, im = d.real, d.imag
        base[i].re = re
        base[i].im = im
      assert size_count == (self.size - 1)
    assert data_count == (self.jobs - 1)

    gpu_fft_execute(self.fft)

    for j in xrange(self.jobs):
      base = self.fft.out + j * self.fft.step;
      for i in xrange(self.size):
        self.buffer[j][i] = base[i].re + base[i].im * 1j


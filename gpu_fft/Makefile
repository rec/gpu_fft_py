HEX_FILES = hex/shader_256.hex \
    hex/shader_512.hex \
    hex/shader_1k.hex \
    hex/shader_2k.hex \
    hex/shader_4k.hex \
    hex/shader_8k.hex \
    hex/shader_16k.hex \
    hex/shader_32k.hex \
    hex/shader_64k.hex \
    hex/shader_128k.hex

SRC_FILES = mailbox.c gpu_fft.c gpu_fft_twiddles.c gpu_fft_shaders.c
BINARY_SRC_FILES = $(SRC_FILES) hello_fft.c

OBJECT_FILES = mailbox.o gpu_fft.o gpu_fft_twiddles.o gpu_fft_shaders.o

BINARY_FILE = hello_fft.bin
ARCHIVE_FILE = libgpufft.a

LINK_FLAGS = -lrt -lm -o $(BINARY_FILE)
ARCHIVE_FLAGS = -rcs $(ARCHIVE_FILE)

$(BINARY_FILE):	$(HEX_FILES) $(BINARY_SRC_FILES)
	gcc $(LINK_FLAGS) $(BINARY_SRC_FILES)

$(ARCHIVE_FILE):	$(HEX_FILES) $(OBJECT_FILES)
	ar $(ARCHIVE_FLAGS) $(OBJECT_FILES)

clean:
	rm -f $(BINARY_FILE) $(OBJECT_FILES) $(ARCHIVE_FILE)

lib: $(ARCHIVE_FILE)
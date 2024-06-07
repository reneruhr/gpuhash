#define STB_IMAGE_IMPLEMENTATION
#include "../extern/stb/stb_image.h"


struct stb_image
{
	int w, h, comp;
	u8* data;
};


stb_image load_image(const char* path)
{
	stb_image img{};
	stbi_set_flip_vertically_on_load(true);  
    img.data =  stbi_load(path, &img.w, &img.h, &img.comp, 4);
	
	printf("Loaded %s. Properties: w=%i, h=%i, c=%i\n", path, img.w, img.h, img.comp);
	return img;
}


void free_image(stb_image img)
{
	stbi_image_free((void*)img.data);
}

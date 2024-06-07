#include <ft2build.h>
#include FT_FREETYPE_H

struct font
{
	vec4_f32 xyuv;
	u32 n;
};

struct character_info 
{
	f32 ax; // advance.x
	f32 ay; // advance.y

	f32 bw; // bitmap.width;
	f32 bh; // bitmap.rows;

	f32 bl; // bitmap_left;
	f32 bt; // bitmap_top;

	f32 tx; // x offset of glyph in texture coordinates
} c[128];

struct font_rendering
{
	opengl_context context;

	u32 atlas_w{};
	u32 atlas_h{};

	const u32 max_letters = 150'000;
	u32 n_strings{};
	
	void add_text(const char* text, f32 x, f32 y, f32 sx, f32 sy, u32 color_id, arena& arena) 
	{
		// https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Text_Rendering_02
		const u32 n_vertices = 6 * strlen(text);
		vec4_f32* vertices = alloc_n<vec4_f32>(arena, n_vertices);
		u32* stringids = alloc_n<u32>(arena, n_vertices);
		u32 u{};
		for (const char* p = text; *p; p++) 
		{
			f32 x2 =  x + c[*p].bl * sx;
			f32 y2 =  y + c[*p].bt * sy;
			f32 w = c[*p].bw * sx;
			f32 h = c[*p].bh * sy;

			x += c[*p].ax * sx;
			y += c[*p].ay * sy;

			if (!w || !h) continue;

			vertices[u++] = {x2,  	y2  , c[*p].tx,				       0                  };
			vertices[u++] = {x2,  	y2-h, c[*p].tx,					   c[*p].bh / atlas_h };
			vertices[u++] = {x2+w,	y2  , c[*p].tx + c[*p].bw / atlas_w, 0                  };
			vertices[u++] = {x2+w,	y2  , c[*p].tx + c[*p].bw / atlas_w, 0				  };
			vertices[u++] = {x2,  	y2-h, c[*p].tx,					   c[*p].bh / atlas_h };
			vertices[u++] = {x2+w,	y2-h, c[*p].tx + c[*p].bw / atlas_w, c[*p].bh / atlas_h };
		}
		
		for (u32 i{}; i<n_vertices; i++) stringids[i] = color_id;

		update_buffer_storage(context.storages[0].buffer,  u*sizeof(vec4_f32), vertices, context.count * sizeof(vec4_f32));
		update_buffer_storage(context.storages[1].buffer,  u*sizeof(u32),     stringids, context.count * sizeof(u32));
		context.count +=u;
		assert(context.count < max_letters && "Font buffer to small");
		n_strings++;
	}
	
	void init_fonts(arena &arena)
	{

		FT_Library ft;
		if(FT_Init_FreeType(&ft)) 
		{
			fprintf(stderr, "Could not init freetype library\n");
		}

		FT_Face face;

		if(FT_New_Face(ft, "extern\\freesans\\FreeSans.ttf", 0, &face)) 
		{
			fprintf(stderr, "Could not open font\n");
		}
		FT_Set_Pixel_Sizes(face, 0, 48);  

		FT_GlyphSlot g = face->glyph;
		FT_Render_Glyph(g, FT_RENDER_MODE_SDF);

		storage_info storages_temp[] =
		{
			{.binding = 0, .size = max_letters * sizeof(vec4_f32) * 6, .data = nullptr} ,
			{.binding = 1, .size = max_letters * sizeof(u32),          .data = nullptr} ,
			{.binding = 2, .size = n_colors * sizeof(vec4_f32),     .data = reinterpret_cast<u32*>(colors)}
		};
		u32 n_storages = sizeof(storages_temp) / sizeof(storage_info);
		storage_info* storages = alloc_n<storage_info>(arena, n_storages);
		memcpy(storages, storages_temp, sizeof(storages_temp));

		context = 	
		{
			.program = compile_shaders(R"(
												#version 460 core 

												struct pos_tex
												{
													vec2 pos;
													vec2 tex;
												};

												layout(std430, binding = 0) restrict readonly buffer vertices 
												{
													pos_tex coord[];
												};

												layout(std430, binding = 1) restrict readonly buffer stringid 
												{
													unsigned int in_stringid[];
												};

												layout(std430, binding = 2) restrict readonly buffer colors
												{
													vec4 in_colors[];
												};

												layout (std140, binding = 0) uniform Transform
												{
													mat4 mvp;
												};

												layout (location = 0) out vec2 texcoord;
												layout (location = 1) out vec4 color;

												void main(void) 
												{

													vec4 pos    = vec4(coord[gl_VertexID].pos,0,1);
													gl_Position = mvp*pos;
													texcoord    = coord[gl_VertexID].tex;
													color       = in_colors[in_stringid[gl_VertexID]];

												}

												)",
												R"(
												#version 460 core 

												layout (location = 0) in vec2 texcoord;
												layout (location = 1) in vec4 color;

												out vec4 frag_color;
												uniform sampler2D tex;

												void main(void) 
												{
													float glyph = texture2D(tex, texcoord).r;
													if (glyph < 0.5) discard;
													frag_color = vec4(1, 1, 1, glyph) * color;
												    float gamma_inverse = 1./2.2;
													frag_color.rgb = pow(frag_color.rgb, vec3(gamma_inverse));
												}

												)"),
			.mode = GL_TRIANGLES,
			.first = 0,
			.count = 0,
			.draw_mode = opengl_context::draw_mode::array,
			.vao   = create_vao(),
			.tex   = 0,
			.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
			.n_storages = n_storages,
			.storages   =   storages
		};

		bind_uniform_block(context.uniform, 0);


		for (u32 u{}; u < n_storages; u++)
		{
			auto& storage = context.storages[u];
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}

		atlas_w = 0;
		atlas_h = 0;


		for(int i = 32; i < 128; i++) 
		{
			if(FT_Load_Char(face, i, FT_LOAD_RENDER)) 
			{
				fprintf(stderr, "Loading character %c failed!\n", i);
				continue;
			}

			atlas_w += g->bitmap.width;
			atlas_h = std::max(atlas_h, g->bitmap.rows);
		}

		glCreateTextures(GL_TEXTURE_2D, 1, &context.tex);
		glTextureParameteri(context.tex, GL_TEXTURE_MAX_LEVEL, 0);
		glTextureParameteri(context.tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(context.tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTextureParameteri(context.tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(context.tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTextureStorage2D(context.tex, 1, GL_R8, atlas_w, atlas_h);
		int x = 0;

		for(u32 i = 32; i < 128; i++) 
		{
			if (FT_Load_Char(face, i, FT_LOAD_RENDER)) continue;
			if(g->bitmap.width)
				glTextureSubImage2D(context.tex, 0, x, 0, g->bitmap.width, g->bitmap.rows, GL_RED, GL_UNSIGNED_BYTE, g->bitmap.buffer);

			c[i].ax = g->advance.x >> 6;
			c[i].ay = g->advance.y >> 6;

			c[i].bw = g->bitmap.width;
			c[i].bh = g->bitmap.rows;

			c[i].bl = g->bitmap_left;
			c[i].bt = g->bitmap_top;

			c[i].tx = (f32)x / atlas_w;

			x += g->bitmap.width;
		}
	}

};

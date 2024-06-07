vec2_u32* square_grid(u32 size = 512, u32 cols = 4, u32 rows = 2, u32 padding = 10, bool up = 1)
{
	u32 n = rows * cols;
	auto p = alloc_n<vec2_u32>(arena0, n);
	
	vec2_u32 total_size = { size * cols, size * rows };

	size += 2*padding;

	for (u32 u{}; u<n;u++)
	{
		if(up)
			p[u] = { size * (u % cols)+padding, size * (u / cols) + padding};
		else
			p[u] = { total_size.x - (size * (u % cols)+padding), total_size.y -  (size * (u / cols) + padding)};
	}
	return p;
}

//Left-child right-sibling binary tree
struct hierachy 
{
	u32 parent;
	u32 next;
	u32 first;
	u32 level;
};

void print(hierachy h)
{
	printf("P: %u. N: %u. F: %u. L: %u\n", h.parent, h.next, h.first, h.level);
}

enum gui_flags
{
	gui_flag_none    = 0,
	gui_flag_unused  = 1 << 0,
	gui_flag_texture = 1 << 1,
	gui_flag_font	 = 1 << 2,
	gui_flag_button  = 1 << 3,
	gui_flag_hover   = 1 << 4,
	gui_flag_sli_add = 1 << 5,
	gui_flag_sli_mul = 1 << 6,
};

struct gui_box
{
	u32 id{};
	u32 flags{};
};

struct gui_data
{
	dynamic_array<gui_box> boxes{&arena1}; 
	dynamic_array<hierachy> nodes{&arena1};
	dynamic_array<vec2_u32> positions{&arena1};
	dynamic_array<vec2_u32> sizes{&arena1};
	dynamic_array<vec4_f32> colors{&arena1};

	std::unordered_map<u32,u32> textures;
	std::unordered_map<u32,std::string> text;
	u32 cur = ~0u;

	u32 max_boxes = 400;
	opengl_context context;
	
	//gpu data
	dynamic_array<vec2_f32> meshes{&arena1};
	dynamic_array<u32>	  indices{&arena1};
	dynamic_array<u32>	  mesh_ids{&arena1};
	dynamic_array<vec2_f32> mesh_uvs{&arena1};
	dynamic_array<per_frame_data> mesh_uniforms{&arena1};

	struct character_info 
	{
		f32 ax; // advance.x
		f32 ay; // advance.y

		f32 bw; // bitmap.width;
		f32 bh; // bitmap.rows;

		f32 bl; // bitmap_left;
		f32 bt; // bitmap_top;

		f32 tx; // x offset of glyph in texture coordinates
	};

	std::unordered_map<u32 /*pixel_size */, std::array<character_info,128>> fonts;
	std::unordered_map<u32 /*pixel_size */, vec2_u32>					    atlas_size;
	std::unordered_map<u32 /*pixel_size */, u32>						    y_offset;

	vec2_u32 max_font_atlas_size{ 1 << 13, 1 << 10 };
	u32 total_font_offset_y = 0;
	const u32 font_padding  = 0;

} gui;


void init_gui_context()
{
	auto program_textured = compile_shaders(R"(
											#version 460 core 

											struct per_frame_data 
											{
												vec4 quat;
												vec3 translation;
												float scale;
												vec4 color;
												unsigned int tex_type;
												float pad[3];
											}; 

											layout(std430, binding = 0) restrict readonly buffer vertices
											{
												vec2 in_vertices[];
											};

											layout(std430, binding = 2) restrict readonly buffer meshids
											{
												unsigned int in_meshid[];
											};

											layout(std430, binding = 3) restrict readonly buffer per_frame_datas
											{
												per_frame_data pfd[];
											};

											layout(std430, binding = 4) restrict readonly buffer texturecoords
											{
												vec2 in_uv[];
											};

											layout (std140, binding = 0) uniform camera 
											{
												mat4 cam;
											};

											layout (location=0) out vec4 frag_color;
											layout (location=1) out vec2 frag_uv;
											layout (location=2) out flat unsigned int tex_type;

											void main(void)
											{
												unsigned int id = in_meshid[gl_VertexID];
												vec3 pos = vec3(pfd[id].scale * in_vertices[gl_VertexID], 0) + pfd[id].translation;
												gl_Position = cam*vec4(pos, 1.0);
												frag_color = pfd[id].color;
												frag_uv = in_uv[gl_VertexID];
												tex_type = pfd[id].tex_type;
											}
											)",
											R"(
											#version 460 core 

											#define gui_flag_texture (1u << 1)
											#define gui_flag_font	 (1u << 2)
											#define gui_flag_button  (1u << 3)
											#define gui_flag_hover   (1u << 4)
											#define gui_flag_sli_add (1u << 5)
											#define gui_flag_sli_mul (1u << 6)


											layout (location=0) in  vec4 frag_color;
											layout (location=1) in  vec2 frag_uv;
											layout (location=2) in flat unsigned int tex_type;
											layout (location=0) out vec4 color;
											//uniform sampler2D texture0;
											layout (binding=0) uniform sampler2D texture_;
											layout (binding=1) uniform sampler2D font_atlas;
											void main(void)
											{
												color = frag_color;
												if((tex_type & gui_flag_texture) != 0)
													color  *= texture(texture_, frag_uv);
												else if((tex_type & gui_flag_font) != 0)
												{
													float glyph = texture2D(font_atlas, frag_uv).r;
													if (glyph < 0.5) discard;
													color *= vec4(1, 1, 1, glyph);
												}
												if((tex_type & gui_flag_button) != 0)
													;	
												if((tex_type & gui_flag_hover) != 0)
													color *= 1.3f;
												float gamma_inverse = 1./2.2;
												color.rgb = pow(color.rgb, vec3(gamma_inverse));
											}
											)");

	

	storage_info storages_temp[] =
	{
	{.binding = 0, .size = gui.max_boxes * 4 * sizeof(vec2_f32), .data = 0},
	{.binding = 2, .size = gui.max_boxes * 6 * sizeof(u32), .data = 0},
	{.binding = 3, .size = gui.max_boxes * 1 * sizeof(per_frame_data) ,.data = 0},
	{.binding = 4, .size = gui.max_boxes * 4 * sizeof(vec2_f32) ,.data = 0},
	};

	u32 n_storages = arraysize(storages_temp);
	storage_info* storages = alloc_n<storage_info>(arena0, n_storages);
	memcpy(storages, storages_temp, sizeof(storages_temp));

	for (u32 u{}; u < n_storages; u++)
	{
		auto& storage = storages[u] ;
		storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
		bind_storage_buffer(storage.buffer, storage.binding);
	}

	gui.context =
	{
		.program = program_textured,
		.mode = GL_TRIANGLES,
		.first = 0,
		.count = 0,
		.draw_mode = opengl_context::draw_mode::elements,
		.vao = create_vao(),
		.ebo    = create_buffer(sizeof(u32)*gui.max_boxes*6),
		.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.n_storages = n_storages,
		.storages = storages
	};
	bind_ebo(&gui.context);


	vec2_u32 gui_max_tex_size = { 1024,1024 };

	glCreateTextures(GL_TEXTURE_2D, 1, &gui.context.tex);
	glTextureParameteri(gui.context.tex, GL_TEXTURE_MAX_LEVEL, 0);
	glTextureParameteri(gui.context.tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(gui.context.tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTextureStorage2D(gui.context.tex, 1, GL_RGBA8, gui_max_tex_size.x, gui_max_tex_size.y);
	glActiveTexture(GL_TEXTURE0);
	glBindTextureUnit(0, gui.context.tex);

};


void pop_tree()
{
	if(gui.cur != ~0u)
	{
		auto& cur_node = gui.nodes[gui.cur];
		if (cur_node.parent != ~0u)
			gui.cur = cur_node.parent;
	}
}

u32 add_child_node()
{
	if (gui.cur != ~0u)
	{
		auto& cur_node = gui.nodes[gui.cur];
		hierachy h = { gui.cur, ~0u,~0u, cur_node.level + 1 };
		if (cur_node.first == ~0u)
		{
			cur_node.first = gui.nodes.size();
			gui.nodes.push_back(h);
		}
		else
		{
			cur_node = gui.nodes[cur_node.first];
			while (cur_node.next != ~0u)
				cur_node = gui.nodes[cur_node.next];
			cur_node.next = gui.nodes.size();
			gui.nodes.push_back(h);
		}
	}
	else
	{
		hierachy h = { ~0u, ~0u,~0u, 0 };
		gui.nodes.push_back(h);
	}
	gui.cur = gui.nodes.size() - 1;
	return gui.nodes.size() - 1;
}

u32 add_next_node()
{
	auto& cur_node = gui.nodes[gui.cur];

	hierachy h = { cur_node.parent, ~0u,~0u, cur_node.level };

	while(cur_node.next != ~0u)
		cur_node = gui.nodes[cur_node.next];
	cur_node.next = gui.nodes.size();
	gui.nodes.push_back(h);
	gui.cur = gui.nodes.size() - 1;
	return gui.nodes.size() - 1;
}

void create_mesh_from_box(u32 id)
{
	assert(gui.boxes.size() < gui.max_boxes);
	assert(gui.context.count + 6 < gui.max_boxes * 6);

	u32 n_vertices = 4;
	f32 a = gui.sizes[id].x;
	f32 b = gui.sizes[id].y;
	f32 x = gui.positions[id].x;
	f32 y = gui.positions[id].y;

	vec2_f32 vertices_[] =
	{
		{x,			y	 },
		{x + a,		y	 },
		{x + a,		y + b},
		{x,			y + b},
	};

	u32 n_indices = 6;
	u32 offset = gui.meshes.size();
	u32 indices_[] = 
	{
		offset+0,offset+1,offset+3, 
		offset+1,offset+2,offset+3
	};

	gui.context.count += 6;

	for(u32 u{}; u<n_vertices; u++)
	{
		gui.meshes.push_back(vertices_[u]);
		gui.mesh_ids.push_back(id);
		gui.mesh_uvs.push_back({ vertices_[u].x==x+a ? 1.f : 0.f, vertices_[u].y==y+b ? 1.f : 0.f });
	}
	
	for (u32 u{}; u < n_indices; u++)
		gui.indices.push_back(indices_[u]);

	per_frame_data uniform{};
	uniform.color = gui.colors[id];
	uniform.texture = gui.boxes[id].flags;
	if (gui.nodes[id].parent != ~0)
	{
		uniform.translation = gui.mesh_uniforms[gui.nodes[id].parent].translation + vec3_f32(0,0,0.001);
	}
	gui.mesh_uniforms.push_back(uniform);
}

u32 add_box(vec2_u32 size, vec2_u32 pos, vec4_f32 color)
{
	u32 id = add_child_node();
	gui.positions.push_back(pos);
	gui.sizes.push_back(size);
	gui.colors.push_back(color);
	gui.boxes.push_back({ id,0 });
	create_mesh_from_box(id);
	return id;
}

u32 next_box(vec2_u32 size, vec2_u32 pos, vec4_f32 color)
{
	u32 id = add_next_node();
	gui.positions.push_back(pos);
	gui.sizes.push_back(size);
	gui.colors.push_back(color);
	gui.boxes.push_back({ id,0 });
	create_mesh_from_box(id);
	return id;
}


void add_image(vec2_u32 size, vec2_u32 pos, u32 tex_id)
{
	u32 id = add_box(size, pos, colors[white]);

	gui.boxes.back().flags |= gui_flag_texture;
	gui.textures[id] = tex_id;
}

void init_font(u32 pixel_size)
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

	
	FT_Set_Pixel_Sizes(face, 0, pixel_size);  

	FT_GlyphSlot g = face->glyph;
	FT_Render_Glyph(g, FT_RENDER_MODE_SDF);


	u32 atlas_w = 0;
	u32 atlas_h = 0;


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
	
	if (!gui.total_font_offset_y)
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &gui.context.font_atlas);
		glTextureParameteri(gui.context.font_atlas, GL_TEXTURE_MAX_LEVEL, 0);
		glTextureParameteri(gui.context.font_atlas, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(gui.context.font_atlas, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTextureParameteri(gui.context.font_atlas, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(gui.context.font_atlas, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTextureStorage2D(gui.context.font_atlas, 1, GL_R8, gui.max_font_atlas_size.x, gui.max_font_atlas_size.y);

		glActiveTexture(GL_TEXTURE1);
		glBindTextureUnit(1, gui.context.font_atlas);
	}

	int x = 0;

	auto& c = gui.fonts[pixel_size];

	
	gui.atlas_size[pixel_size] = { atlas_w, atlas_h }   ;
	gui.y_offset[pixel_size]   = gui.total_font_offset_y;

	for(u32 i = 32; i < 128; i++) 
	{
		if (FT_Load_Char(face, i, FT_LOAD_RENDER)) continue;
		if(g->bitmap.width)
			glTextureSubImage2D(gui.context.font_atlas, 0, x, gui.total_font_offset_y, g->bitmap.width, g->bitmap.rows, GL_RED, GL_UNSIGNED_BYTE, g->bitmap.buffer);

		c[i].ax = g->advance.x >> 6;
		c[i].ay = g->advance.y >> 6;

		c[i].bw = g->bitmap.width;
		c[i].bh = g->bitmap.rows;

		c[i].bl = g->bitmap_left;
		c[i].bt = g->bitmap_top;

		c[i].tx = (f32)x;

		x += g->bitmap.width;
	}
	

	gui.total_font_offset_y   += atlas_h;
	assert(atlas_w                 <= gui.max_font_atlas_size.x);
	assert(gui.total_font_offset_y <= gui.max_font_atlas_size.y);

	FT_Done_Face    (face);
    FT_Done_FreeType(ft)  ;
}

void add_text(u32 id, std::string text, vec4_f32 color)
{
	gui.text[id] = text;
	u32 text_length = gui.text[id].size();

	auto size = gui.sizes[id];
	auto pos = gui.positions[id] + vec2_u32(gui.font_padding, size.y/4);

	const u32 n_vertices = 4 * text_length;
	const u32 n_indices  = 6 * text_length;
	const u32 n_ids      = 1 * text_length;

	vec2_f32* vertices   = alloc_n<vec2_f32>(arena1, n_vertices);
	vec2_f32* uv         = alloc_n<vec2_f32>(arena1, n_vertices);
	u32*	  indices    = alloc_n<u32>(     arena1, n_indices);
	u32*	  ids        = alloc_n<u32>(     arena1, n_vertices);

	u32 font_size = std::max(5u, size.y/2 - gui.font_padding);
	if (!gui.fonts.contains(font_size)) init_font(font_size);

	auto sx{ 1 }, sy{ 1 };
	u32 u{};
	u32 i{};

	u32 offset = gui.meshes.size();

	auto& c				= gui.fonts[font_size];

	vec2_u32 atlas_size = gui.atlas_size[font_size];
	f32      y_offset   = gui.y_offset[font_size];

	u32 m{};
	for (const char* p = text.c_str(); *p; p++)
	{
		u32 next_id = u==0 ? add_child_node() : add_next_node();
		m++;

		f32 x2 = pos.x + c[*p].bl * sx;
		f32 y2 = pos.y + c[*p].bt * sy;
		f32 w = c[*p].bw * sx;
		f32 h = c[*p].bh * sy;

		pos.x += c[*p].ax * sx;
		pos.y += c[*p].ay * sy;

		if (!w || !h) ;

		ids[u] = next_id;  vertices[u] = { x2,  	y2-h};	      uv[u++] = { c[*p].tx    / gui.max_font_atlas_size.x,				(c[*p].bh + y_offset) / gui.max_font_atlas_size.y};
		ids[u] = next_id;  vertices[u] = { x2 + w,	y2-h};        uv[u++] = { (c[*p].tx + c[*p].bw) / gui.max_font_atlas_size.x,	(c[*p].bh + y_offset) / gui.max_font_atlas_size.y};
		ids[u] = next_id;  vertices[u] = { x2 + w,	y2  };	      uv[u++] = { (c[*p].tx + c[*p].bw) / gui.max_font_atlas_size.x,				y_offset  / gui.max_font_atlas_size.y};
		ids[u] = next_id;  vertices[u] = { x2,  	y2  };        uv[u++] = { c[*p].tx    / gui.max_font_atlas_size.x              ,	        y_offset  / gui.max_font_atlas_size.y};
	
		indices[i++] = offset;   indices[i++] = offset+1; indices[i++] = offset+3;
		indices[i++] = offset+1; indices[i++] = offset+2; indices[i++] = offset+3;

		gui.context.count += 6;

		offset += 4;
	}

	assert(m == n_ids);

	for(u32 u{}; u<n_vertices; u++)
	{
		gui.meshes.push_back(vertices[u]);
		gui.mesh_ids.push_back(ids[u]);
		gui.mesh_uvs.push_back(uv[u]);
	}
	
	for (u32 u{}; u < n_indices; u++)
		gui.indices.push_back(indices[u]);

	per_frame_data uniform{};
	uniform.color       = color;
	uniform.translation = gui.mesh_uniforms[id].translation + vec3_f32(0,0,0.1);
	uniform.texture     = gui_flag_font;

	for (u32 u{}; u < n_ids; u++)
	{
		gui.mesh_uniforms.push_back(uniform);
		gui.boxes.push_back({ids[4 * u], gui_flag_font});
		gui.positions.push_back(pos);
		gui.sizes.push_back(size);
		gui.colors.push_back(color);
	}
}

void add_button(vec2_u32 size, vec2_u32 pos, vec4_f32 color, vec4_f32 color_font, std::string text)
{
	u32 id = add_box(size, pos, color);
	gui.boxes.back().flags           = gui_flag_button;
	gui.mesh_uniforms.back().texture = gui_flag_button;

	add_text(id, text, color_font);
	//pop_tree();
}

bool button(vec2_u32 size, vec2_u32 pos, vec4_f32 color, vec4_f32 color_font, std::string text)
{
	add_button(size, pos, color, color_font, text);
	if (!(mouse_state & mouse_state_left_pressed)) return false;
	auto p = mouse_pos;
	if (p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y) return true;
	return false;
}

void add_textbox(vec2_u32 size, vec2_u32 pos, vec4_f32 color, vec4_f32 color_font, std::string text)
{
	//auto timer  = chrono_timer_scoped("\t\ttext_box");
	u32 id = add_box(size, pos, color);
	add_text(id, text, color_font);
	pop_tree();
}

bool toggle(vec2_u32 size, vec2_u32 pos, bool value, vec4_f32 color = colors[white], vec4_f32 color_check = colors[black])
{
	if(button(size, pos, color, color, ""))
		value = !value;
	if (value)
	{
		add_box(0.7f * size, pos+0.15f*size, color_check);
		pop_tree();
	}
	return value;
}

u32 uint_slider(vec2_u32 size, vec2_u32 pos, vec4_f32 color, u32 value, u32 min, u32 max, gui_flags slider_flag, u32 mul = 2 /* optional for gui_flag_sli_mul */)
{
	assert(min <= max);
	assert((slider_flag & gui_flag_sli_add) || ((slider_flag & gui_flag_sli_mul) && min>0 && mul>1) );
	value = std::max(value, min);
	value = std::min(value, max);

	f32 relative_value;

	if (slider_flag & gui_flag_sli_add)
	{
		relative_value = (f32)(value - min) / (max - min);
	}
	else
	{
		f32 k = std::log2((f32)value/ min) / std::log2(mul);
		f32 k_max = std::log2((f32)max / min) / std::log2(mul);
		relative_value = (f32)(k - 0) / (k_max - 0);
	}

	if (mouse_state & (mouse_state_left_pressed | mouse_state_left_keep_pressed))
	{
		auto p = mouse_pos;
		if ((p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y))
		{

			p.x = std::max(p.x, pos.x);
			p.x = std::min(p.x, pos.x + size.x);

			relative_value = (f32)(p.x - pos.x) / size.x;

			if (slider_flag & gui_flag_sli_add)
			{
				value = relative_value * (max - min) + min;
			}
			else
			{
				f32 k_max = std::log2((f32)max / min) / std::log2(mul);
				u32 k = relative_value * k_max;
				value = std::pow(mul,k) * min;
			}
		}
	}

	auto slider_val_as_text = std::to_string(value);
	add_textbox(size, pos, color, colors[black], slider_val_as_text);
	add_button(vec2_u32(size.x/10, size.y), pos + vec2_u32(relative_value*size.x-size.x/20, 0), colors[white], colors[black], "");
	
	pop_tree();

	return value;
}


void upload_gui_data()
{
	//auto timer = chrono_timer_scoped("\t\tupload gui");
	auto& storages = gui.context.storages;
	
	map_buffer(gui.indices.data(),  sizeof(u32)*gui.indices.size(), gui.context.ebo, 0);
	update_buffer_storage(storages[0].buffer, gui.meshes.size()   * sizeof(vec2_f32), gui.meshes  .data());
	update_buffer_storage(storages[1].buffer, gui.mesh_ids.size() * sizeof(u32),	  gui.mesh_ids.data());
	update_buffer_storage(storages[2].buffer, gui.mesh_uniforms.size() * sizeof(per_frame_data), gui.mesh_uniforms.data());
	update_buffer_storage(storages[3].buffer, gui.mesh_uvs.size() * sizeof(vec2_f32), gui.mesh_uvs.data());
}

void check_gui()
{
	u32 n = gui.boxes.size();
	assert(n == gui.nodes.size());
	assert(n == gui.positions.size());
	assert(n == gui.sizes.size());
	assert(n == gui.colors.size());

	assert(4*n == gui.meshes.size());
	assert(4*n == gui.mesh_ids.size());
	assert(4*n == gui.mesh_uvs.size());
	assert(n == gui.mesh_uniforms.size());
	assert(gui.context.count == gui.indices.size());
	assert(6*n == gui.indices.size());
}

void clear_gui()
{
	
	gui.boxes.clear();
	gui.nodes.clear();
	gui.positions.clear();
	gui.sizes.clear();
	gui.colors.clear();
	gui.textures.clear();
	gui.text.clear();
	gui.cur = ~0u;

	//gpu data
	gui.meshes.clear();
	gui.indices.clear();
	gui.mesh_ids.clear();
	gui.mesh_uvs.clear();
	gui.mesh_uniforms.clear();
		
	gui.context.count = 0;
}


u32 intersect_gui(vec2_u32 p, gui_flags flag = gui_flag_button, u32 offset = 0)
{
	u32 n = gui.boxes.size();
	for(u32 u = offset ; u<n; u++)
	{
		if (flag && !(gui.boxes[u].flags & flag)) continue;
		auto pos  = gui.positions[u];
		auto size = gui.sizes[u];
		if (p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y) 
		{
			return u;
		}
	}
	return ~0u;
}

void update_hover_gui(u32 id)
{
	if(id != ~0u) gui.mesh_uniforms[id].texture |= gui_flag_hover;
}



void draw_gui()
{
	//auto timer = chrono_timer_scoped("\tdraw gui");
	for (u32 u{}; u < gui.context.n_storages; u++) 
		bind_storage_buffer(gui.context.storages[u].buffer, gui.context.storages[u].binding);

	auto cam = cur_cam();
	map_buffer((void*)cam.m,   sizeof(mat4_f32), gui.context.uniform);
	bind_uniform_block(gui.context.uniform, 0);
	draw(&gui.context);
}

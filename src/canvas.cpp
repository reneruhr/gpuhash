struct graph_canvas
{
	opengl_context context_quads;
	opengl_context context_shapes;
	opengl_context context_lines;
	opengl_context context_points;

	vec2_f32 pos{};
	vec2_f32 size{};
	vec2_f32 graph_offset{};
	vec2_f32 bg_offset{};

	const u32 max_quads    =  1000;
	const u32 max_plots    = 2'000;
	const u32 max_pointset =   200;
	const u32 max_shapes   =   200;

	const u32 max_vertices =    10'000;
	const u32 max_lines    =   150'000;
	const u32 max_points   = 1'000'000;
	const vec2_u32 max_texture_size = { 1024 * 10, 2*1024 };

	u32 n_graphs{};
	u32 n_quads{};
	u32 n_shapes{};
	u32 n_pointset{};
	u32 n_points{};
	u32 n_vertices{};
	u32 n_texel_width{};

	struct per_frame_data
	{
		vec3_f32 translation{0.};
		f32  scale{1};
		vec4_f32 color{1.};
	};	
	struct per_frame_data2
	{
		quat q{ 1. };
		vec3_f32 translation{0.};
		f32  scale{1};
		vec4_f32 color{1.};
		u32 texture{ 0 };
		vec3_f32 pad;
	};	

	enum plot_type
	{
		regular,
		loglog
	};

	void add_graph(f32 *x, f32 *y, u32 n, plot_type type,  vec2_f32 range, vec2_f32 min, vec2_f32 offset, u32 color_id, arena& arena)
	{
			mesh graph{ .vertices = alloc_n<f32>(arena, 2 * (n-1) * 2), .n_vertices = (n-1) * 2, .colors = alloc_n<f32>(arena, 4 * (n-1) * 2) };

			if (type == loglog)
			{
				for (u32 k{}; k < n - 1; k++)
				{
					graph.vertices[4 * k + 0] = ( x[k] > 0 ? std::log(x[k  ]/min.x) / range.x * size.x : 0 ) + offset.x;
					graph.vertices[4 * k + 1] = ( y[k] > 0 ? std::log(y[k  ]/min.y) / range.y * size.y : 0 ) + offset.y;
					graph.vertices[4 * k + 2] = ( x[k] > 0 ? std::log(x[k+1]/min.x) / range.x * size.x : 0 ) + offset.x;
					graph.vertices[4 * k + 3] = ( y[k] > 0 ? std::log(y[k+1]/min.y) / range.y * size.y : 0 ) + offset.y;
				}
				vec4_f32 color = colors[color_id];
				fill(graph.colors, 2 * (n - 1), &color.x);
				add_line_mesh(graph, arena);
			}
			else if(type == regular)
			{
				for (u32 k{}; k < n - 1; k++)
				{
					graph.vertices[4 * k + 0] =  std::max(x[k], min.x) / range.x * size.x + offset.x;
					graph.vertices[4 * k + 1] =  std::max(y[k], min.y) / range.y * size.y + offset.y;
					graph.vertices[4 * k + 2] =  std::max(x[k+1], min.x) / range.x * size.x + offset.x;
					graph.vertices[4 * k + 3] =  std::max(y[k+1], min.y) / range.y * size.y + offset.y;
				}
				vec4_f32 color = colors[color_id];
				fill(graph.colors, 2 * (n - 1), &color.x);
				add_line_mesh(graph, arena);
			}
	}

	void add_line_mesh(mesh line_plot, f32 height, arena& arena)
	{
		assert(n_graphs < max_plots - 1);
		u32* ids = alloc_n<u32>(arena, line_plot.n_vertices);
		std::fill(ids, ids + line_plot.n_vertices, n_graphs);
		per_frame_data pfd{};
		pfd.translation.z += height;
		update_buffer_storage(context_lines.storages[0].buffer, 2 * sizeof(f32) * line_plot.n_vertices, line_plot.vertices, 2 * sizeof(f32) * context_lines.count);
		update_buffer_storage(context_lines.storages[1].buffer, 4 * sizeof(f32) * line_plot.n_vertices, line_plot.colors,   4 * sizeof(f32) * context_lines.count);
		update_buffer_storage(context_lines.storages[2].buffer, 1 * sizeof(u32) * line_plot.n_vertices,	ids,                1 * sizeof(u32) * context_lines.count);
		update_buffer_storage(context_lines.storages[3].buffer, sizeof(per_frame_data),                 &pfd,			    1 * sizeof(per_frame_data) * n_graphs);

		n_graphs++;
		context_lines.count += line_plot.n_vertices;
	}

	void add_line_mesh(mesh line_plot, arena& arena)
	{
		add_line_mesh(line_plot, 0.1, arena);
	}

	void add_background(f32 x, f32 y, arena& arena)
	{
		assert(n_quads < max_quads-1);
		auto rect = rectangle(size.x + 4*size.x / 10 * bg_offset.x, size.y + 4*size.y / 10 * bg_offset.y, arena);
		for (u32 u{}; u < rect.n_vertices; u++)
		{
			rect.vertices[2 * u]     += x;
			rect.vertices[2 * u + 1] += y;
		};
		for (u32 u{}; u < rect.n_indices; u++)
			rect.indices[u]  += n_quads*4;

		u32 id[] = { n_quads, n_quads, n_quads, n_quads };
		per_frame_data pfd{};
		map_buffer(rect.indices,  1*sizeof(u32)*6, context_quads.ebo,                n_quads*6*sizeof(u32));
		update_buffer_storage(context_quads.storages[0].buffer, 2*sizeof(f32)*4,		rect.vertices, n_quads*2*sizeof(f32)*4);
		update_buffer_storage(context_quads.storages[1].buffer, 4*sizeof(f32)*4,		rect.colors,   n_quads*4*sizeof(f32)*4);
		update_buffer_storage(context_quads.storages[2].buffer, 1*sizeof(u32)*4,		id,            n_quads*1*sizeof(u32)*4);
		update_buffer_storage(context_quads.storages[3].buffer, sizeof(per_frame_data),&pfd,           n_quads*sizeof(per_frame_data));
		context_quads.count += 6;
		n_quads++;
	}

	void add_legend(f32 x, f32 y, vec2_f32 size, arena& arena)
	{
		assert(n_quads < max_quads-1);
		auto rect = rectangle(size.x, size.y, arena);
		for (u32 u{}; u < rect.n_vertices; u++)
		{
			rect.vertices[2 * u]     += x;
			rect.vertices[2 * u + 1] += y;
		};
		for (u32 u{}; u < rect.n_indices; u++)
			rect.indices[u]  += n_quads*4;
		
		u32 id[] = { n_quads, n_quads, n_quads, n_quads };
		per_frame_data pfd{};
		pfd.translation.z += 0.5;
		pfd.color = colors[Puce];
		map_buffer(rect.indices,  1*sizeof(u32)*6, context_quads.ebo,                n_quads*6*sizeof(u32));
		update_buffer_storage(context_quads.storages[0].buffer, 2*sizeof(f32)*4,		rect.vertices, n_quads*2*sizeof(f32)*4);
		update_buffer_storage(context_quads.storages[1].buffer, 4*sizeof(f32)*4,		rect.colors,   n_quads*4*sizeof(f32)*4);
		update_buffer_storage(context_quads.storages[2].buffer, 1*sizeof(u32)*4,		id,            n_quads*1*sizeof(u32)*4);
		update_buffer_storage(context_quads.storages[3].buffer, sizeof(per_frame_data),&pfd,          n_quads*sizeof(per_frame_data));
		context_quads.count += 6;
		n_quads++;
	}

	void add_shape(mesh mesh, vec3_f32 pos, f32 scale, arena& arena)
	{
		assert(n_shapes  < max_shapes-1);
		assert(n_vertices+mesh.n_vertices < max_vertices);
		assert(context_shapes.count+mesh.n_indices < 6*max_vertices);

		u32* indices = alloc_n<u32>(arena, mesh.n_indices);
		for (u32 u{}; u < mesh.n_indices; u++)
			indices[u]  = mesh.indices[u]+n_vertices;
		
		u32* ids = alloc_n<u32>(arena, mesh.n_vertices);
		std::fill(ids, ids + mesh.n_vertices, n_shapes);
		per_frame_data2 pfd{};
		pfd.translation = pos;
		pfd.scale       = scale;
		pfd.color       = colors[white];
		pfd.texture		= mesh.uv != nullptr ? 1 : 0;
		map_buffer(indices,  sizeof(u32)*mesh.n_indices, context_shapes.ebo, context_shapes.count*sizeof(u32));
		update_buffer_storage(context_shapes.storages[0].buffer, mesh.n_vertices*sizeof(f32)*2,		mesh.vertices, n_vertices*2*sizeof(f32));
		update_buffer_storage(context_shapes.storages[1].buffer, mesh.n_vertices*sizeof(f32)*4,		mesh.colors,   n_vertices*4*sizeof(f32));
		update_buffer_storage(context_shapes.storages[2].buffer, mesh.n_vertices*sizeof(u32),		ids,           n_vertices*1*sizeof(u32));
		update_buffer_storage(context_shapes.storages[3].buffer, sizeof(per_frame_data2),            &pfd,		   n_shapes*sizeof(per_frame_data2));
		if(mesh.uv != nullptr)
		{
			vec2_f32* uv= alloc_n<vec2_f32>(arena, mesh.n_vertices);
			for (u32 u{}; u < mesh.n_vertices; u++)
				uv[u] = { (mesh.uv[2*u] * mesh.texture.w + n_texel_width) / max_texture_size.x, mesh.uv[2*u+1] * mesh.texture.h / max_texture_size.y};
			update_buffer_storage(context_shapes.storages[4].buffer, mesh.n_vertices*sizeof(f32)*2,		uv, n_vertices*2*sizeof(f32));
			add_texture(mesh.texture);
		}
		context_shapes.count += mesh.n_indices;
		n_vertices += mesh.n_vertices;
		n_shapes++;
	}

	void add_points(points pts, vec3_f32 pos, f32 scale, arena& arena)
	{
		assert(n_pointset < max_pointset -1 );
		assert(context_points.count+pts.n_vertices < max_points);

		u32* ids = alloc_n<u32>(arena, pts.n_vertices);
		std::fill(ids, ids + pts.n_vertices, n_pointset);
		per_frame_data pfd{};
		pfd.translation = pos;
		pfd.scale       = scale;

		vec4_f32 *colors_ = alloc_n<vec4_f32>(arena, pts.n_vertices);
		std::fill(colors_, colors_ + pts.n_vertices, colors[Red]);

		update_buffer_storage(context_points.storages[0].buffer, pts.n_vertices*sizeof(f32)*2,		pts.vertices, context_points.count*2*sizeof(f32));
		update_buffer_storage(context_points.storages[1].buffer, pts.n_vertices*sizeof(f32)*4,		colors_,      context_points.count*4*sizeof(f32));
		update_buffer_storage(context_points.storages[2].buffer, pts.n_vertices*sizeof(u32),		ids,          context_points.count*1*sizeof(u32));
		update_buffer_storage(context_points.storages[3].buffer, sizeof(per_frame_data),            &pfd,		  n_pointset*sizeof(per_frame_data));
		context_points.count += pts.n_vertices;
		n_pointset++;
	}

	void add_tags(f32* x, f32*y, u32 nx, u32 ny, vec2_f32 offset, arena& arena)
	{
		mesh tags;
		f32 tag_length   = 10;
		tags.n_vertices  = 2*(nx+ny);
		tags.vertices    = alloc_n<f32>(arena, 2*tags.n_vertices);
		tags.colors      = alloc_n<f32>(arena, 4*tags.n_vertices);

		for (u32 u{}; u<nx; u++)
		{
			tags.vertices[4 * u] = x[u]*size.x + offset.x;
			tags.vertices[4 * u+1] = tag_length + offset.y;
			tags.vertices[4 * u+2] = x[u]*size.x + offset.x;
			tags.vertices[4 * u+3] = -tag_length + offset.y;
		}
		for (u32 u{nx}; u<nx+ny; u++)
		{
			tags.vertices[4 * u]   = -tag_length + offset.x;
			tags.vertices[4 * u+1] = y[u-nx]*size.y + offset.y;
			tags.vertices[4 * u+2] =  tag_length + offset.x;
			tags.vertices[4 * u+3] = y[u-nx]*size.y + offset.y;
		}

		fill(tags.colors, tags.n_vertices, &colors[black].x);
		add_line_mesh(tags, arena);
	}

	void add_axis(f32 x, f32 y, arena& arena)
	{
		mesh axis;
		axis.n_vertices = 4;
		axis.vertices    = alloc_n<f32>(arena, 2*4);
		axis.vertices[0] = x;
		axis.vertices[1] = y;
		axis.vertices[2] = x+size.x;
		axis.vertices[3] = y;
		axis.vertices[4] = x;
		axis.vertices[5] = y;
		axis.vertices[6] = x;
		axis.vertices[7] = y+size.y;

		axis.colors= alloc_n<f32>(arena, 4*4);
		fill(axis.colors, 4, &colors[black].x);
		add_line_mesh(axis, arena);
	}

	graph_canvas() = default;

	graph_canvas(vec2_f32 size, vec2_f32 pos, arena& arena) : size(size), pos(pos)
	{
		graph_offset = vec2_f32(0.2f, 0.2f);
		bg_offset = vec2_f32(0.4f, 0.4f);

		auto program = compile_shaders(R"(
											#version 460 core 

											struct per_frame_data 
											{
												vec3 translation;
												float scale;
												vec4 color;
											}; 

											layout(std430, binding = 0) restrict readonly buffer vertices
											{
												vec2 in_vertices[];
											};

											layout(std430, binding = 1) restrict readonly buffer colors
											{
												vec4 in_colors[];
											};

											layout(std430, binding = 2) restrict readonly buffer meshids
											{
												unsigned int in_meshid[];
											};

											layout(std430, binding = 3) restrict readonly buffer per_frame_datas
											{
												per_frame_data pfd[];
											};

											layout (std140, binding = 0) uniform camera 
											{
												mat4 cam;
											};

											layout (location=0) out vec4 frag_color;

											void main(void)
											{
												unsigned int id = in_meshid[gl_VertexID];
												vec3 pos = vec3(pfd[id].scale * in_vertices[gl_VertexID], 0) + pfd[id].translation;
												gl_Position = cam*vec4(pos, 1.0);
												frag_color = pfd[id].color * in_colors[gl_VertexID];	
											}
											)",
											R"(
											#version 460 core 
											layout (location=0) in  vec4 frag_color;
											layout (location=0) out vec4 color;
											void main(void)
											{
												color = frag_color;
												float gamma_inverse = 1./2.2;
												color.rgb = pow(color.rgb, vec3(gamma_inverse));
											}
											)");



		auto program_textured = compile_shaders(R"(
											#version 460 core 

											struct per_frame_data2 
											{
												vec4 quat;
												vec3 translation;
												float scale;
												vec4 color;
												unsigned int has_texture;
												float pad[3];
											}; 

											layout(std430, binding = 0) restrict readonly buffer vertices
											{
												vec2 in_vertices[];
											};

											layout(std430, binding = 1) restrict readonly buffer colors
											{
												vec4 in_colors[];
											};

											layout(std430, binding = 2) restrict readonly buffer meshids
											{
												unsigned int in_meshid[];
											};

											layout(std430, binding = 3) restrict readonly buffer per_frame_datas
											{
												per_frame_data2 pfd[];
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
											layout (location=2) out flat unsigned int has_tex;

											void main(void)
											{
												unsigned int id = in_meshid[gl_VertexID];
												vec3 pos = vec3(pfd[id].scale * in_vertices[gl_VertexID], 0) + pfd[id].translation;
												gl_Position = cam*vec4(pos, 1.0);
												frag_color = pfd[id].color * in_colors[gl_VertexID];	
												frag_uv = in_uv[gl_VertexID];
												has_tex = pfd[id].has_texture;
											}
											)",
											R"(
											#version 460 core 
											layout (location=0) in  vec4 frag_color;
											layout (location=1) in  vec2 frag_uv;
											layout (location=2) in flat unsigned int has_tex;
											layout (location=0) out vec4 color;
											uniform sampler2D texture0;
											void main(void)
											{
												if(bool(has_tex)) 
													color = frag_color*texture(texture0, frag_uv);
												else
													color = frag_color;
												float gamma_inverse = 1./2.2;
												color.rgb = pow(color.rgb, vec3(gamma_inverse));
											}
											)");

		storage_info storages_temp_lines[] =
		{
		{.binding = 0, .size = max_lines * 4 * sizeof(f32)*2, .data = 0},
		{.binding = 1, .size = max_lines * 4 * sizeof(f32)*4, .data = 0},
		{.binding = 2, .size = max_lines * 4 * sizeof(u32)*1, .data = 0},
		{.binding = 3, .size = max_plots * 1 * sizeof(graph_canvas::per_frame_data) ,.data = 0},
		};

		u32 n_storages_lines = sizeof(storages_temp_lines) / sizeof(storage_info);
		storage_info* storages_lines = alloc_n<storage_info>(arena, n_storages_lines);
		memcpy(storages_lines, storages_temp_lines, sizeof(storages_temp_lines));

		for (u32 u{}; u < n_storages_lines; u++)
		{
			auto& storage = storages_lines[u] ;
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}

		context_lines =
		{
			.program = program,
			.mode = GL_LINES,
			.first = 0,
			.count = 0,
			.draw_mode = opengl_context::draw_mode::array,
			.vao = create_vao(),
			.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
			.n_storages = n_storages_lines,
			.storages = storages_lines
		};


		storage_info storages_temp[] =
		{
		{.binding = 0, .size = max_quads * 4 * sizeof(f32)*2, .data = 0},
		{.binding = 1, .size = max_quads * 4 * sizeof(f32)*4, .data = 0},
		{.binding = 2, .size = max_quads * 4 * sizeof(u32)*1, .data = 0},
		{.binding = 3, .size = max_quads * 1 * sizeof(graph_canvas::per_frame_data) ,.data = 0},
		};

		u32 n_storages = sizeof(storages_temp) / sizeof(storage_info);
		storage_info* storages = alloc_n<storage_info>(arena, n_storages);
		memcpy(storages, storages_temp, sizeof(storages_temp));

		context_quads =
		{
		.program = program,
		.mode = GL_TRIANGLES,
		.first = 0,
		.count = 0,
		.draw_mode = opengl_context::draw_mode::elements,
		.vao = create_vao(),
		.ebo = create_buffer(sizeof(u32) * 6 * max_quads),
		.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.n_storages = n_storages,
		.storages = storages
		};

		bind_ebo(&context_quads);

		for (u32 u{}; u < n_storages; u++)
		{
			auto& storage = context_quads.storages[u];
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}


		storage_info storages_temp_points[] =
		{
		{.binding = 0, .size = max_points * 4 * sizeof(f32)*2, .data = 0},
		{.binding = 1, .size = max_points * 4 * sizeof(f32)*4, .data = 0},
		{.binding = 2, .size = max_points * 4 * sizeof(u32)*1, .data = 0},
		{.binding = 3, .size = max_pointset * 1 * sizeof(graph_canvas::per_frame_data) ,.data = 0},
		};

		u32 n_storages_points = sizeof(storages_temp_points) / sizeof(storage_info);
		storage_info* storages_points = alloc_n<storage_info>(arena, n_storages_points);
		memcpy(storages_points, storages_temp_points, sizeof(storages_temp_points));

		for (u32 u{}; u < n_storages_points; u++)
		{
			auto& storage = storages_points[u] ;
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}

		context_points =
		{
			.program = program,
			.mode = GL_POINTS,
			.first = 0,
			.count = 0,
			.draw_mode = opengl_context::draw_mode::array,
			.vao = create_vao(),
			.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
			.n_storages = n_storages_points,
			.storages = storages_points
		};


		storage_info storages_temp_shapes[] =
		{
		{.binding = 0, .size = max_vertices *  sizeof(f32)*2, .data = 0},
		{.binding = 1, .size = max_vertices *  sizeof(f32)*4, .data = 0},
		{.binding = 2, .size = max_vertices *  sizeof(u32)*1, .data = 0},
		{.binding = 3, .size = max_shapes   *  sizeof(graph_canvas::per_frame_data2) ,.data = 0},
		{.binding = 4, .size = max_vertices *  sizeof(f32)*2 ,.data = 0},
		};

		u32 n_storages_shapes = sizeof(storages_temp_shapes) / sizeof(storage_info);
		storage_info* storages_shapes = alloc_n<storage_info>(arena, n_storages_shapes);
		memcpy(storages_shapes, storages_temp_shapes, sizeof(storages_temp_shapes));

		context_shapes =
		{
		.program = program_textured,
		.mode = GL_TRIANGLES,
		.first = 0,
		.count = 0,
		.draw_mode = opengl_context::draw_mode::elements,
		.vao = create_vao(),
		.ebo = create_buffer(sizeof(u32) * 6 * max_vertices),
		.uniform = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.n_storages = n_storages_shapes,
		.storages = storages_shapes
		};


		bind_ebo(&context_shapes);

		for (u32 u{}; u < n_storages_shapes; u++)
		{
			auto& storage = context_shapes.storages[u];
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}


	glCreateTextures(GL_TEXTURE_2D, 1, &context_shapes.tex);
	glTextureParameteri(context_shapes.tex, GL_TEXTURE_MAX_LEVEL, 0);
	glTextureParameteri(context_shapes.tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(context_shapes.tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureStorage2D(context_shapes.tex, 1, GL_RGBA8, max_texture_size.x, max_texture_size.y);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTextures(0, 1, &context_shapes.tex);


	}


	void add_texture(image image)
	{
		assert(n_texel_width + image.w < max_texture_size.x);
		assert(image.h < max_texture_size.y);
		glTextureSubImage2D(context_shapes.tex, 0, n_texel_width, 0, image.w, image.h, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
		n_texel_width += image.w;
	}

	void draw_canvas(const mat4_f32& cam)
	{
		for (u32 u{}; u < context_quads.n_storages; u++) 
			bind_storage_buffer(context_quads.storages[u].buffer, context_quads.storages[u].binding);
		auto cam0 = mul(cam, translation(vec3_f32(pos.x-2*size.x/10*bg_offset.x, pos.y-2*size.y/10*bg_offset.y, 0)));
		map_buffer((void*)cam0.m,   sizeof(mat4_f32), context_quads.uniform);
		bind_uniform_block(context_quads.uniform, 0);
		draw(&context_quads);

		for (u32 u{}; u < context_lines.n_storages; u++) 
			bind_storage_buffer(context_lines.storages[u].buffer, context_lines.storages[u].binding);
		auto cam1 = mul(cam, translation(vec3_f32(pos.x, pos.y, 0)));
		map_buffer((void*)cam1.m,   sizeof(mat4_f32), context_lines.uniform);
		bind_uniform_block(context_lines.uniform, 0);
		draw(&context_lines);

		for (u32 u{}; u < context_shapes.n_storages; u++) 
			bind_storage_buffer(context_shapes.storages[u].buffer, context_shapes.storages[u].binding);
		map_buffer((void*)cam.m,   sizeof(mat4_f32), context_shapes.uniform);
		bind_uniform_block(context_shapes.uniform, 0);
		glBindTextures(0, 1, &context_shapes.tex);
		draw(&context_shapes);

		for (u32 u{}; u < context_points.n_storages; u++) 
			bind_storage_buffer(context_points.storages[u].buffer, context_points.storages[u].binding);
		map_buffer((void*)cam.m,   sizeof(mat4_f32), context_points.uniform);
		bind_uniform_block(context_points.uniform, 0);
		draw(&context_points);
	}

	void fill(f32* cs, u32 n, f32* c)
	{
		for (u32 u{}; u < n; u++)
		{
			cs[4 * u + 0] = c[0];
			cs[4 * u + 1] = c[1];
			cs[4 * u + 2] = c[2];
			cs[4 * u + 3] = c[3];
		}
	}
};

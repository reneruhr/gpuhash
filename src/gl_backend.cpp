//#define GLFW_INCLUDE_GLEXT
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <cassert>
#include "common.h"
#include "vec_math.h"


struct storage_info
{
	u32 buffer;
	u32 binding;
	u32 size;
	u32 *data;
};

struct opengl_context
{
	GLuint program;

	GLenum mode;
	GLenum first;
	GLenum count;
	enum class draw_mode { array, elements} draw_mode;

	GLuint vao;
	GLuint vbo;
	GLuint vbo2;
	GLuint vbo3;
	GLuint ebo;

	GLuint tex;
	GLuint font_atlas;
	vec2_u32 tex_size;

	GLuint uniform;
	GLuint uniform2;

	u32 n_storages;
	storage_info *storages;

	u32 n_work_groups_x;
	u32 n_work_groups_y;
	u32 n_work_groups_z;
	u32 local_size_x;
	u32 local_size_y;
	u32 local_size_z;
};

struct image
{
	u8* data;
	u32 size;
	u32 w;
	u32 h;
};

struct mesh
{
	f32 *vertices{};
	u32 n_vertices{};
	u32 *indices{};
	u32 n_indices{};
	f32 *colors{};
	f32 *uv;
	u32* mesh_indices{};
	image texture{};
};

struct points
{
	f32 *vertices{};
	u32 n_vertices{};
};

struct points_u32
{
	u32 *vertices{};
	u32 n_vertices{};
};


struct mvp 
{
	mat4_f32 m{1.f};
	mat4_f32 v{1.f};
	mat4_f32 p{1.f};

	f32* data() { return static_cast<f32*>(m.m); }
};




// Vertex Array structures 
GLuint create_vao();
void   create_vertexarray(opengl_context*, u32 location = 0, u32 size = 4);
void   bind_ebo(opengl_context* context);

// Shaders 
GLuint compile_shaders(const GLchar* frag, const GLchar* vert);
GLuint compile_shaders(const GLchar* comp);


// Mutable data buffers
GLuint create_buffer(u32 size, GLenum mode = GL_STATIC_DRAW);
void   map_buffer(void *data, u32 size, GLuint buffer, GLuint offset = 0);

// Immutable data buffers
void   bind_uniform_block(u32 uniform, u32 bind_point);
void   bind_storage_buffer(u32 buffer, u32 bind_point);



GLFWwindow* create_window(u32 w, u32 h, const char* title)
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(w, h, title, NULL, NULL);
    glfwMakeContextCurrent(window);

    int version = gladLoadGL(glfwGetProcAddress);
    printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

	return window;
}



void draw(opengl_context *context)
{
	glBindVertexArray(context->vao);
	glUseProgram(context->program);

	if (context->draw_mode == opengl_context::draw_mode::array)
	{
		glDrawArrays(context->mode, context->first, context->count);
	}
	else if (context->draw_mode == opengl_context::draw_mode::elements)
	{
		glDrawElements(context->mode, context->count, GL_UNSIGNED_INT, 0);
	}
}

void draw_indexed(opengl_context *context, u32 first, u32 count)
{
	glBindVertexArray(context->vao);
	glUseProgram(context->program);

	if (context->draw_mode == opengl_context::draw_mode::array)
	{
		glDrawArrays(context->mode, first, count);
	}
	else if (context->draw_mode == opengl_context::draw_mode::elements)
	{
		glDrawElements(context->mode, count, GL_UNSIGNED_INT, (void*)+first);
	}
}

void create_vertexarray(opengl_context *context, u32 location, u32 size)
{
	u32 binding = location;
	glEnableVertexArrayAttrib(context->vao, location);
	glVertexArrayAttribFormat(context->vao, location, size, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(context->vao, location, binding );
	glVertexArrayVertexBuffer(context->vao, binding, location == 0 ? context->vbo : context->vbo2, 0, size*sizeof(f32));
}

void upload_points(opengl_context& context, vec4_f32 *pos, u32 n_vertices)
{
	create_vertexarray(&context);
	map_buffer(pos, 4 * sizeof(f32) * n_vertices, context.vbo);
}

void bind_uniform_block(GLuint uniform, u32 bind_point = 0)
{
	glBindBufferBase(GL_UNIFORM_BUFFER, bind_point, uniform);
}

void map_buffer(void *data, u32 size, GLuint buffer, u32 offset)
{
	void* gpu_handle = glMapNamedBufferRange(buffer, (GLintptr)offset, (GLsizeiptr)size, GL_MAP_WRITE_BIT);
	memcpy(gpu_handle, data, size);
	glUnmapNamedBuffer(buffer);
}

GLuint create_buffer(u32 size, GLenum mode)
{
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferData(handle, size, 0, mode);
	return {handle};
}


GLuint buffer_storage(u32 size, void  *data, GLenum mode)
{
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferStorage(handle, size, data, mode);
	return {handle};
}

void update_buffer_storage(GLuint storage, u32 size, void *data, u32 offset = 0)
{
	glNamedBufferSubData(storage, offset , size, data);
}

void bind_storage_buffer(GLuint buffer, u32 bind_point = 0)
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_point, buffer);
}

void* gpu_to_cpu_persistent(u32 buffer, u32 size)
{
	u32 scratch_buffer = buffer_storage(size, nullptr, GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_READ_BIT);
	glCopyNamedBufferSubData(buffer, scratch_buffer, 0, 0, size);
	auto p = glMapNamedBufferRange(	scratch_buffer, 0, size, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	return p;
}

void gpu_to_cpu(u32 buffer, void* cpu_buffer, u32 size)
{
	glGetNamedBufferSubData(buffer, 0, size, cpu_buffer);
}


GLuint create_vao()
{
	GLuint handle;
	glCreateVertexArrays(1, &handle);
	return {handle};
}

void bind_ebo(opengl_context* context)
{
	if(context->draw_mode == opengl_context::draw_mode::elements)
		glVertexArrayElementBuffer(context->vao, context->ebo);
}










struct opengl_profiler
{
	const u32 lag_query{8};
	const u32 n_queries{16};
	u32 queries[16];
	u32 current_query{};

	u32 n_data{};
	u64* data{};
};

void init(opengl_profiler* p)
{
	glGenQueries(p->n_queries, p->queries);
}

// https://registry.khronos.org/OpenGL/extensions/ARB/ARB_timer_query.txt	
void profile_compute(opengl_context *context, opengl_profiler *p)
{
	u32& u = p->current_query;
	glUseProgram(context->program);

	glBeginQuery(GL_TIME_ELAPSED, p->queries[u % p->n_queries]);

	glDispatchCompute(context->n_work_groups_x, context->n_work_groups_y, context->n_work_groups_z);

	glEndQuery(GL_TIME_ELAPSED);

	if (u >= p->lag_query)
	{
		GLint available{};
		while (!available)
		{
			glGetQueryObjectiv(p->queries[(u - p->lag_query) % p->n_queries], GL_QUERY_RESULT_AVAILABLE, &available);
		}

		u64 ns_elapsed{};
		glGetQueryObjectui64v(p->queries[(u - p->lag_query) % p->n_queries], GL_QUERY_RESULT, &ns_elapsed);
		if (u < p->n_data + p->lag_query)
			p->data[u - p->lag_query] = ns_elapsed;
	}

	u++;
}
















enum class shader_log { shader, program };

void compile_info(GLuint handle, const char* debug_name, shader_log type){
    char buffer[8192];
	GLsizei length = 0;
	if(type == shader_log::shader )
		glGetShaderInfoLog(handle, sizeof(buffer), &length, buffer);
	else
		glGetProgramInfoLog(handle, sizeof(buffer), &length, buffer);
    if (length)
	{
		printf("%s (File: %s)\n", buffer, debug_name);
		assert(false);
	}
};

GLuint compile_shaders(const GLchar* vert_shader_src, const GLchar* frag_shader_src)
{
	GLuint vert_shader;
	GLuint frag_shader;
	GLuint program;

	vert_shader = glCreateShader(GL_VERTEX_SHADER);
	frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vert_shader, 1, &vert_shader_src, 0);
	compile_info(vert_shader, "Vertex shader", shader_log::shader);
	glShaderSource(frag_shader, 1, &frag_shader_src, 0);
	compile_info(frag_shader, "Fragment shader", shader_log::shader);
	glCompileShader(vert_shader);
	glCompileShader(frag_shader);
	program = glCreateProgram();
	glAttachShader(program, vert_shader);
	glAttachShader(program, frag_shader);
	glLinkProgram(program);
	compile_info(program, "Program", shader_log::program);
	
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);

	return program;
}

GLuint compile_shaders(const GLchar* comp_shader_src)
{
	GLuint comp_shader;
	GLuint program;

	comp_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(comp_shader, 1, &comp_shader_src, 0);
	compile_info(comp_shader, "Compute shader", shader_log::shader);
	glCompileShader(comp_shader);
	program = glCreateProgram();
	glAttachShader(program, comp_shader);
	glLinkProgram(program);
	compile_info(program, "Program", shader_log::program);

	glDeleteShader(comp_shader);

	return program;
}













struct opengl_constants
{
	int shared_memory_size; 
	int work_group_count_x; 
	int work_group_count_y; 
	int work_group_count_z; 
	int local_size; 
	int local_size_x; 
	int local_size_y; 
	int local_size_z; 

	int storage_block_size;
};


void query_opengl_constants(opengl_constants& constants)
{
/*
void GetBooleanv( enum pname, boolean *data );
void GetIntegerv( enum pname, int *data );
void GetInteger64v( enum pname, int64 *data );
void GetFloatv( enum pname, float *data );
void GetDoublev( enum pname, double *data );
*/

glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &constants.shared_memory_size);

glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0, &constants.work_group_count_x);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,1, &constants.work_group_count_y);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,2, &constants.work_group_count_z);


glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &constants.local_size);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0, &constants.local_size_x);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,1, &constants.local_size_y);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,2, &constants.local_size_z);


glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &constants.storage_block_size);

}












opengl_context context_for_points(u32 n_vertices)
{
	return 
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 pos;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											void main(void)
											{
												gl_Position = p*(v*(m*pos));
											}
											)", 
											R"(
											#version 460 core
											out vec4 color;
											void main(void)
											{
												color = vec4(62./255, 207./255, 173./255, 1);
											}
											)"),
		.mode			= GL_POINTS,
		.first			= 0,
		.count			= n_vertices,
		.draw_mode		= opengl_context::draw_mode::array,
		.vao			= create_vao(),
		.vbo			= create_buffer(4*sizeof(f32)*n_vertices),
		.vbo2			= 0,
		.ebo			= 0,
		.uniform		= create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};
}

opengl_context context_for_2d_points(points p)
{
	opengl_context context_points
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec2 pos;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											out vec4 frag_color;
											void main(void)
											{
												gl_Position = p*(v*(m*vec4(pos,0,1)));
												frag_color = vec4(0,0,0,1);
	
											}
											)", 
											R"(
											#version 460 core
											in vec4 frag_color;
											out vec4 color;
											void main(void)
											{
												color = frag_color;
											}
											)"),
		.mode			= GL_POINTS,
		.first			= 0,
		.count   = p.n_vertices,
		.draw_mode		= opengl_context::draw_mode::array,
		.vao     = create_vao(),
		.vbo     = create_buffer(2 * sizeof(f32) * p.n_vertices),
		.uniform = create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};


	create_vertexarray(&context_points, 0,2);
	map_buffer(p.vertices, 2 * sizeof(f32) * p.n_vertices, context_points.vbo);

	return context_points;
}

opengl_context context_from_colored_mesh_2d(mesh mesh)
{
	opengl_context context_mesh
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec2 pos;
											layout (location = 1) in vec4 col;
											layout (std140, binding = 0) uniform Camera 
												{
												mat4 cam;
												};
											out vec4 frag_color;
											void main(void)
											{
												gl_Position = cam*vec4(pos,0,1);
												frag_color = col;
											}
											)", 
											R"(
											#version 460 core
											in vec4 frag_color;
											out vec4 color;
											void main(void)
											{
												color = frag_color;
											}
											)"),
		.mode			= GL_TRIANGLES,
		.first			= 0,
		.count  = mesh.n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao	  = create_vao(),
		.vbo	  = create_buffer(2*sizeof(f32)*mesh.n_vertices),
		.vbo2	  = create_buffer(4*sizeof(f32)*mesh.n_vertices),
		.ebo    = create_buffer(sizeof(u32)*mesh.n_indices),
		.uniform= create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};

	bind_ebo(&context_mesh);
	create_vertexarray(&context_mesh, 0,2);
	create_vertexarray(&context_mesh, 1);

	map_buffer(mesh.vertices, 2*sizeof(f32)*mesh.n_vertices, context_mesh.vbo);
	map_buffer(mesh.colors  , 4*sizeof(f32)*mesh.n_vertices, context_mesh.vbo2);
	map_buffer(mesh.indices,   sizeof(u32)*mesh.n_indices, context_mesh.ebo);

	return context_mesh;
}

opengl_context context_from_textured_mesh(mesh mesh)
{
	opengl_context context_mesh
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 pos;
											layout (location = 1) in vec2 uv;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											out vec2 frag_uv;
											void main(void)
											{
												gl_Position = p*(v*(m*pos));
												frag_uv= uv;
											}
											)", 
											R"(
											#version 460 core
											in vec2 frag_uv;
											uniform sampler2D texture0;
											out vec4 color;
											void main(void)
											{
												color = texture(texture0,frag_uv);
											}
											)"),
		.mode			= GL_TRIANGLES,
		.first			= 0,
		.count  = mesh.n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao	= create_vao(),
		.vbo	= create_buffer(4*sizeof(f32)*mesh.n_vertices),
		.vbo2	= create_buffer(2*sizeof(f32)*mesh.n_vertices),
		.ebo    = create_buffer(sizeof(u32)*mesh.n_indices),
		.uniform= create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};


	bind_ebo(&context_mesh);
	create_vertexarray(&context_mesh, 0);
	create_vertexarray(&context_mesh, 1,2);

	map_buffer(mesh.vertices, 4*sizeof(f32)*mesh.n_vertices, context_mesh.vbo);
	map_buffer(mesh.uv,       2*sizeof(f32)*mesh.n_vertices, context_mesh.vbo2);
	map_buffer(mesh.indices,   sizeof(u32)*mesh.n_indices, context_mesh.ebo);

	return context_mesh;
}



using func_2d =  f32(*)(f32,f32);

image draw(func_2d f, u32 n, arena& arena)
{
	u8* data = alloc_n<u8>(arena, n*n*4);
	for (u32 i{}; i < n; i++)
		for (u32 j{}; j < n; j++)
		{
			f32 x = 2*(f32)i / (n-1) - 1.;
			f32 y = 2*(f32)j / (n-1) - 1.;

			f32 r = x * x + y * y;
			u8 fval = 255u * f(x, y);
			data[4 * (j + i * n  )] = r <= 1.f ? fval : 0;
			data[4*(j + i * n) + 1] = r <= 1.f ? fval : 0;
			data[4*(j + i * n) + 2] = r <= 1.f ? fval : 0;
			data[4*(j + i * n) + 3] = r <= 1.f ? 255u : 0;
		}
	return { data, n * n , n, n};
}

opengl_context context_from_func_2d(mesh mesh,func_2d f, u32 size, arena& arena)
{
	opengl_context context_mesh
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 pos;
											layout (location = 1) in vec2 uv;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											out vec2 frag_uv;
											void main(void)
											{
												gl_Position = p*(v*(m*pos));
												frag_uv= uv;
											}
											)", 
											R"(
											#version 460 core
											in vec2 frag_uv;
											uniform sampler2D texture0;
											out vec4 color;
											void main(void)
											{
												color = texture(texture0,frag_uv);
											}
											)"),
		.mode			= GL_TRIANGLES,
		.first			= 0,
		.count  = mesh.n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao	= create_vao(),
		.vbo	= create_buffer(4*sizeof(f32)*mesh.n_vertices),
		.vbo2	= create_buffer(2*sizeof(f32)*mesh.n_vertices),
		.ebo    = create_buffer(sizeof(u32)*mesh.n_indices),
		.uniform= create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};


	bind_ebo(&context_mesh);
	create_vertexarray(&context_mesh, 0);
	create_vertexarray(&context_mesh, 1,2);

	map_buffer(mesh.vertices, 4*sizeof(f32)*mesh.n_vertices, context_mesh.vbo);
	map_buffer(mesh.uv,       2*sizeof(f32)*mesh.n_vertices, context_mesh.vbo2);
	map_buffer(mesh.indices,   sizeof(u32)*mesh.n_indices, context_mesh.ebo);

	image image = draw(f, size, arena);

	glCreateTextures(GL_TEXTURE_2D, 1, &context_mesh.tex);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MAX_LEVEL, 0);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureStorage2D(context_mesh.tex, 1, GL_RGBA8, size, size);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTextureSubImage2D(context_mesh.tex, 0, 0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
	glBindTextures(0, 1, &context_mesh.tex);
	return context_mesh;
}


#include "load_image.cpp"

opengl_context context_from_image_2d(mesh mesh, stb_image img)
{
	opengl_context context_mesh
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 pos;
											layout (location = 1) in vec2 uv;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											out vec2 frag_uv;
											void main(void)
											{
												gl_Position = p*(v*(m*pos));
												frag_uv= uv;
											}
											)", 
											R"(
											#version 460 core
											in vec2 frag_uv;
											uniform sampler2D texture0;
											out vec4 color;
											void main(void)
											{
												color = texture(texture0,frag_uv);
											}
											)"),
		.mode			= GL_TRIANGLES,
		.first			= 0,
		.count  = mesh.n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao	= create_vao(),
		.vbo	= create_buffer(4*sizeof(f32)*mesh.n_vertices),
		.vbo2	= create_buffer(2*sizeof(f32)*mesh.n_vertices),
		.ebo    = create_buffer(sizeof(u32)*mesh.n_indices),
		.uniform= create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};


	bind_ebo(&context_mesh);
	create_vertexarray(&context_mesh, 0);
	create_vertexarray(&context_mesh, 1,2);

	map_buffer(mesh.vertices, 4*sizeof(f32)*mesh.n_vertices, context_mesh.vbo);
	map_buffer(mesh.uv,       2*sizeof(f32)*mesh.n_vertices, context_mesh.vbo2);
	map_buffer(mesh.indices,   sizeof(u32)*mesh.n_indices, context_mesh.ebo);


	glCreateTextures(GL_TEXTURE_2D, 1, &context_mesh.tex);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MAX_LEVEL, 0);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(context_mesh.tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureStorage2D(context_mesh.tex, 1, GL_RGBA8, img.w, img.h);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTextureSubImage2D(context_mesh.tex, 0, 0, 0, img.w, img.h, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
	glBindTextures(0, 1, &context_mesh.tex);
	return context_mesh;
}


















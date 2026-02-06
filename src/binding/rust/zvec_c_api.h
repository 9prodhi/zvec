// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ZVEC_C_API_H
#define ZVEC_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Core types
// ============================================================================

/// Status returned from API functions. code=0 means success.
typedef struct {
    int32_t code;
    const char* msg;  // NULL on success, points to thread-local error on failure
} ZvecStatus;

/// Non-owning string view (caller owns data)
typedef struct {
    const char* data;
    size_t len;
} ZvecStrView;

/// Owning string (must be freed with zvec_str_free)
typedef struct {
    char* data;
    size_t len;
} ZvecStr;

/// Non-owning view into array data
typedef struct {
    const void* data;
    size_t len;  // Number of elements (not bytes)
} ZvecArrayView;

// ============================================================================
// Opaque handles (pointers to C++ objects)
// ============================================================================

typedef struct ZvecCollection ZvecCollection;
typedef struct ZvecDoc ZvecDoc;
typedef struct ZvecDocList ZvecDocList;
typedef struct ZvecDocMap ZvecDocMap;
typedef struct ZvecFieldSchema ZvecFieldSchema;
typedef struct ZvecCollectionSchema ZvecCollectionSchema;
typedef struct ZvecIndexParams ZvecIndexParams;
typedef struct ZvecQueryParams ZvecQueryParams;
typedef struct ZvecVectorQuery ZvecVectorQuery;
typedef struct ZvecGroupByVectorQuery ZvecGroupByVectorQuery;
typedef struct ZvecStatusList ZvecStatusList;
typedef struct ZvecGroupResults ZvecGroupResults;
typedef struct ZvecGroupResult ZvecGroupResult;

// ============================================================================
// Enums (matching zvec::DataType, etc.)
// ============================================================================

/// Data types for fields
typedef enum {
    ZVEC_DATA_TYPE_UNDEFINED = 0,
    ZVEC_DATA_TYPE_BINARY = 1,
    ZVEC_DATA_TYPE_STRING = 2,
    ZVEC_DATA_TYPE_BOOL = 3,
    ZVEC_DATA_TYPE_INT32 = 4,
    ZVEC_DATA_TYPE_INT64 = 5,
    ZVEC_DATA_TYPE_UINT32 = 6,
    ZVEC_DATA_TYPE_UINT64 = 7,
    ZVEC_DATA_TYPE_FLOAT = 8,
    ZVEC_DATA_TYPE_DOUBLE = 9,
    ZVEC_DATA_TYPE_VECTOR_BINARY32 = 20,
    ZVEC_DATA_TYPE_VECTOR_BINARY64 = 21,
    ZVEC_DATA_TYPE_VECTOR_FP16 = 22,
    ZVEC_DATA_TYPE_VECTOR_FP32 = 23,
    ZVEC_DATA_TYPE_VECTOR_FP64 = 24,
    ZVEC_DATA_TYPE_VECTOR_INT4 = 25,
    ZVEC_DATA_TYPE_VECTOR_INT8 = 26,
    ZVEC_DATA_TYPE_VECTOR_INT16 = 27,
    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 = 30,
    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 = 31,
} ZvecDataType;

/// Metric types for similarity search
typedef enum {
    ZVEC_METRIC_UNDEFINED = 0,
    ZVEC_METRIC_L2 = 1,
    ZVEC_METRIC_IP = 2,
    ZVEC_METRIC_COSINE = 3,
    ZVEC_METRIC_MIPSL2 = 4,
} ZvecMetricType;

/// Quantization types
typedef enum {
    ZVEC_QUANTIZE_UNDEFINED = 0,
    ZVEC_QUANTIZE_FP16 = 1,
    ZVEC_QUANTIZE_INT8 = 2,
    ZVEC_QUANTIZE_INT4 = 3,
} ZvecQuantizeType;

/// Index types
typedef enum {
    ZVEC_INDEX_UNDEFINED = 0,
    ZVEC_INDEX_HNSW = 1,
    ZVEC_INDEX_IVF = 3,
    ZVEC_INDEX_FLAT = 4,
    ZVEC_INDEX_INVERT = 10,
} ZvecIndexType;

/// Operator types for document operations
typedef enum {
    ZVEC_OPERATOR_INSERT = 0,
    ZVEC_OPERATOR_UPSERT = 1,
    ZVEC_OPERATOR_UPDATE = 2,
    ZVEC_OPERATOR_DELETE = 3,
} ZvecOperator;

// ============================================================================
// Collection options
// ============================================================================

typedef struct {
    bool read_only;
    bool enable_mmap;
    uint32_t max_buffer_size;
} ZvecCollectionOptions;

// ============================================================================
// Error handling (thread-local)
// ============================================================================

/// Get the last error message (thread-local). Returns "unknown error" if none.
const char* zvec_last_error_message(void);

/// Free an owning string returned from API
void zvec_str_free(ZvecStr s);

// ============================================================================
// Field Schema APIs
// ============================================================================

/// Create a field schema.
/// @param name Field name
/// @param data_type Field data type
/// @param nullable Whether field can be null
/// @param dimension Vector dimension (0 for non-vector fields)
ZvecFieldSchema* zvec_field_schema_create(
    ZvecStrView name,
    uint32_t data_type,
    bool nullable,
    uint32_t dimension
);

/// Free a field schema
void zvec_field_schema_free(ZvecFieldSchema* fs);

/// Set index parameters on a field schema (for vector fields)
ZvecStatus zvec_field_schema_set_index_params(
    ZvecFieldSchema* fs,
    const ZvecIndexParams* params
);

// ============================================================================
// Collection Schema APIs
// ============================================================================

/// Create a collection schema
ZvecCollectionSchema* zvec_collection_schema_create(ZvecStrView name);

/// Free a collection schema
void zvec_collection_schema_free(ZvecCollectionSchema* cs);

/// Add a field to the schema
ZvecStatus zvec_collection_schema_add_field(
    ZvecCollectionSchema* cs,
    const ZvecFieldSchema* fs
);

/// Set max document count per segment
ZvecStatus zvec_collection_schema_set_max_doc_count_per_segment(
    ZvecCollectionSchema* cs,
    uint64_t count
);

// ============================================================================
// Index Parameters APIs
// ============================================================================

/// Create HNSW index parameters
/// @param metric Metric type (ZVEC_METRIC_*)
/// @param m Number of neighbors per node
/// @param ef_construction ef parameter during construction
/// @param quantize Quantization type (ZVEC_QUANTIZE_*)
ZvecIndexParams* zvec_index_params_hnsw_create(
    uint32_t metric,
    int32_t m,
    int32_t ef_construction,
    uint32_t quantize
);

/// Create IVF index parameters
/// @param metric Metric type
/// @param n_list Number of inverted lists
/// @param n_iters Number of k-means iterations
/// @param use_soar Use SOAR optimization
/// @param quantize Quantization type
ZvecIndexParams* zvec_index_params_ivf_create(
    uint32_t metric,
    int32_t n_list,
    int32_t n_iters,
    bool use_soar,
    uint32_t quantize
);

/// Create Flat index parameters
/// @param metric Metric type
/// @param quantize Quantization type
ZvecIndexParams* zvec_index_params_flat_create(
    uint32_t metric,
    uint32_t quantize
);

/// Create Invert index parameters (for scalar fields)
/// @param range_optimized Optimize for range queries
/// @param extended_wildcard Support extended wildcard
ZvecIndexParams* zvec_index_params_invert_create(
    bool range_optimized,
    bool extended_wildcard
);

/// Free index parameters
void zvec_index_params_free(ZvecIndexParams* params);

// ============================================================================
// Query Parameters APIs
// ============================================================================

/// Create HNSW query parameters
/// @param ef ef parameter for search
/// @param radius Search radius (0 for no limit)
/// @param is_linear Use linear search
/// @param is_using_refiner Use refiner for better accuracy
ZvecQueryParams* zvec_query_params_hnsw_create(
    int32_t ef,
    float radius,
    bool is_linear,
    bool is_using_refiner
);

/// Create IVF query parameters
/// @param n_probes Number of lists to probe
/// @param is_using_refiner Use refiner for better accuracy
/// @param scale_factor Scale factor for search
ZvecQueryParams* zvec_query_params_ivf_create(
    int32_t n_probes,
    bool is_using_refiner,
    float scale_factor
);

/// Create Flat query parameters
/// @param is_using_refiner Use refiner for better accuracy
/// @param scale_factor Scale factor for search
ZvecQueryParams* zvec_query_params_flat_create(
    bool is_using_refiner,
    float scale_factor
);

/// Free query parameters
void zvec_query_params_free(ZvecQueryParams* params);

// ============================================================================
// Collection Lifecycle APIs
// ============================================================================

/// Create and open a new collection
/// @param path Directory path for collection storage
/// @param schema Collection schema
/// @param opts Collection options (can be NULL for defaults)
/// @param out Output pointer to collection handle
ZvecStatus zvec_collection_create_and_open(
    ZvecStrView path,
    const ZvecCollectionSchema* schema,
    const ZvecCollectionOptions* opts,
    ZvecCollection** out
);

/// Open an existing collection
/// @param path Directory path for collection storage
/// @param opts Collection options (can be NULL for defaults)
/// @param out Output pointer to collection handle
ZvecStatus zvec_collection_open(
    ZvecStrView path,
    const ZvecCollectionOptions* opts,
    ZvecCollection** out
);

/// Close and free a collection handle
void zvec_collection_free(ZvecCollection* coll);

/// Destroy a collection (delete all data)
ZvecStatus zvec_collection_destroy(ZvecCollection* coll);

/// Flush pending writes to disk
ZvecStatus zvec_collection_flush(ZvecCollection* coll);

/// Optimize the collection (compact and rebuild indexes)
/// @param concurrency Number of threads to use (0 for default)
ZvecStatus zvec_collection_optimize(ZvecCollection* coll, int32_t concurrency);

/// Get collection path
ZvecStatus zvec_collection_path(const ZvecCollection* coll, ZvecStr* out);

/// Get collection statistics as string
ZvecStatus zvec_collection_stats_string(const ZvecCollection* coll, ZvecStr* out);

/// Get collection schema as string
ZvecStatus zvec_collection_schema_string(const ZvecCollection* coll, ZvecStr* out);

/// Get collection options
ZvecStatus zvec_collection_options(const ZvecCollection* coll, ZvecCollectionOptions* out);

/// Create index on a field
ZvecStatus zvec_collection_create_index(
    ZvecCollection* coll,
    ZvecStrView field_name,
    const ZvecIndexParams* params,
    int32_t concurrency
);

/// Drop index on a field
ZvecStatus zvec_collection_drop_index(
    ZvecCollection* coll,
    ZvecStrView field_name
);

/// Add a new column to the collection
/// @param coll Collection handle
/// @param column Column name
/// @param field Field schema for the new column
/// @param expression Expression to populate the column
/// @param concurrency Number of threads to use
ZvecStatus zvec_collection_add_column(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecFieldSchema* field,
    ZvecStrView expression,
    int32_t concurrency
);

/// Drop a column from the collection
ZvecStatus zvec_collection_drop_column(
    ZvecCollection* coll,
    ZvecStrView column
);

/// Alter a column's schema or rename it
/// @param coll Collection handle
/// @param column Current column name
/// @param new_name New name for the column (empty to keep current)
/// @param new_schema New schema (NULL to keep current)
/// @param concurrency Number of threads to use
ZvecStatus zvec_collection_alter_column(
    ZvecCollection* coll,
    ZvecStrView column,
    ZvecStrView new_name,
    const ZvecFieldSchema* new_schema,
    int32_t concurrency
);

// ============================================================================
// Document APIs
// ============================================================================

/// Create a new empty document
ZvecDoc* zvec_doc_create(void);

/// Free a document
void zvec_doc_free(ZvecDoc* doc);

/// Set the primary key
void zvec_doc_set_pk(ZvecDoc* doc, ZvecStrView pk);

/// Get the primary key (returns owned string, must free)
ZvecStr zvec_doc_get_pk(const ZvecDoc* doc);

/// Get the score (for search results)
float zvec_doc_get_score(const ZvecDoc* doc);

/// Get the document ID
uint64_t zvec_doc_get_doc_id(const ZvecDoc* doc);

/// Set the operator for this document
void zvec_doc_set_operator(ZvecDoc* doc, uint32_t op);

/// Check if a field exists in the document
bool zvec_doc_has_field(const ZvecDoc* doc, ZvecStrView field);

/// Set a bool field
ZvecStatus zvec_doc_set_bool(ZvecDoc* doc, ZvecStrView field, bool value);

/// Set an i32 field
ZvecStatus zvec_doc_set_i32(ZvecDoc* doc, ZvecStrView field, int32_t value);

/// Set an i64 field
ZvecStatus zvec_doc_set_i64(ZvecDoc* doc, ZvecStrView field, int64_t value);

/// Set a u32 field
ZvecStatus zvec_doc_set_u32(ZvecDoc* doc, ZvecStrView field, uint32_t value);

/// Set a u64 field
ZvecStatus zvec_doc_set_u64(ZvecDoc* doc, ZvecStrView field, uint64_t value);

/// Set an f32 field
ZvecStatus zvec_doc_set_f32(ZvecDoc* doc, ZvecStrView field, float value);

/// Set an f64 field
ZvecStatus zvec_doc_set_f64(ZvecDoc* doc, ZvecStrView field, double value);

/// Set a string field
ZvecStatus zvec_doc_set_string(ZvecDoc* doc, ZvecStrView field, ZvecStrView value);

/// Set a dense f32 vector field
ZvecStatus zvec_doc_set_dense_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const float* data,
    size_t len
);

/// Set a dense f64 vector field
ZvecStatus zvec_doc_set_dense_f64(
    ZvecDoc* doc,
    ZvecStrView field,
    const double* data,
    size_t len
);

/// Set a sparse f32 vector field
ZvecStatus zvec_doc_set_sparse_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const uint32_t* indices,
    const float* values,
    size_t len
);

/// Get a string field (returns owned string, must free)
ZvecStatus zvec_doc_get_string(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecStr* out
);

/// Get a string field (alternate name for compatibility)
ZvecStatus zvec_doc_get_string_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecStr* out
);

/// Get a dense vector field
/// @param doc Document
/// @param field Field name
/// @param data_type Output data type
/// @param out Output array view
ZvecStatus zvec_doc_get_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* data_type,
    ZvecArrayView* out
);

/// Get a sparse vector field
/// @param doc Document
/// @param field Field name
/// @param data_type Output data type
/// @param indices_out Output indices array view
/// @param values_out Output values array view
ZvecStatus zvec_doc_get_sparse_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* data_type,
    ZvecArrayView* indices_out,
    ZvecArrayView* values_out
);

/// Get field type information
/// @param doc Document
/// @param field Field name
/// @param found Output: whether field was found
/// @param is_null Output: whether field is null
/// @param data_type Output: field data type
ZvecStatus zvec_doc_get_field_type(
    const ZvecDoc* doc,
    ZvecStrView field,
    bool* found,
    bool* is_null,
    uint32_t* data_type
);

/// Get a dense f32 vector field (returns view into document data)
ZvecStatus zvec_doc_get_dense_f32(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecArrayView* out
);

/// Get an i64 field
ZvecStatus zvec_doc_get_i64(const ZvecDoc* doc, ZvecStrView field, int64_t* out);

/// Get an f64 field
ZvecStatus zvec_doc_get_f64(const ZvecDoc* doc, ZvecStrView field, double* out);

/// Get a bool field
ZvecStatus zvec_doc_get_bool(const ZvecDoc* doc, ZvecStrView field, bool* out);

/// Convert document to string representation
ZvecStr zvec_doc_to_string(const ZvecDoc* doc);

/// Convert document to detailed string representation
ZvecStr zvec_doc_to_detail_string(const ZvecDoc* doc);

// ============================================================================
// Batch Insert/Update APIs
// ============================================================================

/// Insert documents into collection
/// @param coll Collection handle
/// @param docs Array of document pointers
/// @param count Number of documents
/// @param out Output status list (one status per document)
ZvecStatus zvec_collection_insert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Upsert documents (insert or update)
ZvecStatus zvec_collection_upsert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Update existing documents
ZvecStatus zvec_collection_update(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Delete documents by primary keys
ZvecStatus zvec_collection_delete(
    ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecStatusList** out
);

/// Delete documents by filter expression
ZvecStatus zvec_collection_delete_by_filter(
    ZvecCollection* coll,
    ZvecStrView filter
);

/// Fetch documents by primary keys
ZvecStatus zvec_collection_fetch(
    const ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecDocMap** out
);

// ============================================================================
// Vector Query APIs
// ============================================================================

/// Create a vector query for dense or hybrid vectors
/// @param field Vector field name
/// @param data Dense query vector data (bytes)
/// @param len Dense query vector size in bytes
/// @param sparse_indices Sparse vector indices (NULL for dense-only)
/// @param sparse_values Sparse vector values (NULL for dense-only)
/// @param sparse_len Number of sparse elements
/// @param topk Number of results to return
ZvecVectorQuery* zvec_vector_query_create(
    ZvecStrView field,
    const void* data,
    size_t len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    int32_t topk
);

/// Free a vector query
void zvec_vector_query_free(ZvecVectorQuery* query);

/// Set filter expression on query
void zvec_vector_query_set_filter(ZvecVectorQuery* query, ZvecStrView filter);

/// Set query parameters
void zvec_vector_query_set_query_params(
    ZvecVectorQuery* query,
    const ZvecQueryParams* params
);

/// Set whether to include vectors in results
void zvec_vector_query_set_include_vector(ZvecVectorQuery* query, bool include);

/// Set whether to include document ID in results
void zvec_vector_query_set_include_doc_id(ZvecVectorQuery* query, bool include);

/// Set output fields (NULL for all fields)
void zvec_vector_query_set_output_fields(
    ZvecVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
);

/// Execute vector query
ZvecStatus zvec_collection_query(
    const ZvecCollection* coll,
    const ZvecVectorQuery* query,
    ZvecDocList** out
);

// ============================================================================
// Group By Vector Query APIs
// ============================================================================

/// Create a group-by vector query
/// @param field Vector field name
/// @param data Dense query vector data (bytes)
/// @param len Dense query vector size in bytes
/// @param sparse_indices Sparse vector indices (NULL for dense-only)
/// @param sparse_values Sparse vector values (NULL for dense-only)
/// @param sparse_len Number of sparse elements
/// @param group_by_field Field to group results by
/// @param group_count Maximum number of groups
/// @param group_topk Maximum results per group
ZvecGroupByVectorQuery* zvec_group_by_vector_query_create(
    ZvecStrView field,
    const void* data,
    size_t len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    ZvecStrView group_by_field,
    uint32_t group_count,
    uint32_t group_topk
);

/// Free a group-by vector query
void zvec_group_by_vector_query_free(ZvecGroupByVectorQuery* query);

/// Set filter expression on group-by query
void zvec_group_by_vector_query_set_filter(ZvecGroupByVectorQuery* query, ZvecStrView filter);

/// Set output fields for group-by query
void zvec_group_by_vector_query_set_output_fields(
    ZvecGroupByVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
);

/// Set query parameters for group-by query
void zvec_group_by_vector_query_set_query_params(
    ZvecGroupByVectorQuery* query,
    const ZvecQueryParams* params
);

/// Set whether to include vectors in group-by results
void zvec_group_by_vector_query_set_include_vector(ZvecGroupByVectorQuery* query, bool include);

/// Execute group-by vector query
ZvecStatus zvec_collection_group_by_query(
    const ZvecCollection* coll,
    const ZvecGroupByVectorQuery* query,
    ZvecGroupResults** out
);

// ============================================================================
// Result List Accessors (DocList - for query results)
// ============================================================================

/// Get number of documents in list
size_t zvec_doc_list_len(const ZvecDocList* list);

/// Get document at index.
///
/// @warning Returns pointer to thread-local storage. Valid only until next
/// call to this function from same thread. For iteration, prefer direct
/// accessors: zvec_doc_list_get_pk(), zvec_doc_list_get_score(), etc.
const ZvecDoc* zvec_doc_list_get(const ZvecDocList* list, size_t index);

/// Get primary key at index (returns owned string, must free)
ZvecStr zvec_doc_list_get_pk(const ZvecDocList* list, size_t index);

/// Get score at index
float zvec_doc_list_get_score(const ZvecDocList* list, size_t index);

/// Get dense f32 vector at index (returns view into list data)
ZvecStatus zvec_doc_list_get_dense_f32(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    ZvecArrayView* out
);

/// Get string field at index (returns owned string, must free)
ZvecStatus zvec_doc_list_get_string(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    ZvecStr* out
);

/// Get i64 field at index
ZvecStatus zvec_doc_list_get_i64(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    int64_t* out
);

/// Free document list
void zvec_doc_list_free(ZvecDocList* list);

// ============================================================================
// Result Map Accessors (DocMap - for fetch results)
// ============================================================================

/// Get number of documents in map
size_t zvec_doc_map_len(const ZvecDocMap* map);

/// Get key (primary key) at index (returns owned string, must free)
ZvecStr zvec_doc_map_get_key(const ZvecDocMap* map, size_t index);

/// Get document value at index.
///
/// @warning Same thread-local aliasing as zvec_doc_list_get(). Returns pointer
/// to thread-local storage. Valid only until next call to this function from
/// same thread.
const ZvecDoc* zvec_doc_map_get_value(const ZvecDocMap* map, size_t index);

/// Get document at index (alias for get_value)
const ZvecDoc* zvec_doc_map_get_doc(const ZvecDocMap* map, size_t index);

/// Get dense f32 vector at index (returns view into map data)
ZvecStatus zvec_doc_map_get_dense_f32(
    const ZvecDocMap* map,
    size_t index,
    ZvecStrView field,
    ZvecArrayView* out
);

/// Get string field at index (returns owned string, must free)
ZvecStatus zvec_doc_map_get_string(
    const ZvecDocMap* map,
    size_t index,
    ZvecStrView field,
    ZvecStr* out
);

/// Free document map
void zvec_doc_map_free(ZvecDocMap* map);

// ============================================================================
// Status List Accessors (for batch operation results)
// ============================================================================

/// Get number of statuses in list
size_t zvec_status_list_len(const ZvecStatusList* list);

/// Get status at index
ZvecStatus zvec_status_list_get(const ZvecStatusList* list, size_t index);

/// Free status list
void zvec_status_list_free(ZvecStatusList* list);

// ============================================================================
// Group Results Accessors (for group-by query results)
// ============================================================================

/// Get number of groups
size_t zvec_group_results_len(const ZvecGroupResults* results);

/// Get group at index (returns borrowed reference)
const ZvecGroupResult* zvec_group_results_get(const ZvecGroupResults* results, size_t index);

/// Free group results
void zvec_group_results_free(ZvecGroupResults* results);

/// Get the group value (the field value used for grouping)
ZvecStr zvec_group_result_get_group_value(const ZvecGroupResult* result);

/// Get the documents in this group.
///
/// @warning Same thread-local aliasing as zvec_doc_list_get(). Returns pointer
/// to thread-local storage. Valid only until next call to this function from
/// same thread.
const ZvecDocList* zvec_group_result_get_docs(const ZvecGroupResult* result);

// ============================================================================
// SIMD Distance Functions
// ============================================================================

/// Compute squared L2 (Euclidean) distance between two f32 vectors.
/// Uses SIMD-optimized implementation from zvec core.
/// @param a First vector
/// @param b Second vector
/// @param dim Vector dimension
/// @return Squared L2 distance
float zvec_compute_l2_distance(const float* a, const float* b, size_t dim);

/// Compute inner product distance between two f32 vectors.
/// Uses SIMD-optimized implementation from zvec core.
/// @param a First vector
/// @param b Second vector
/// @param dim Vector dimension
/// @return Inner product value
float zvec_compute_ip_distance(const float* a, const float* b, size_t dim);

/// Compute cosine distance between two f32 vectors.
/// Uses SIMD-optimized implementation from zvec core.
/// Note: input vectors should be pre-normalized with norms appended.
/// @param a First vector
/// @param b Second vector
/// @param dim Vector dimension (including extra norm elements)
/// @return Cosine distance (1 - cosine_similarity)
float zvec_compute_cosine_distance(const float* a, const float* b, size_t dim);

/// Compute squared L2 distance from a query vector to N database vectors (batch).
/// Each database vector is contiguous in memory: vectors[i*dim .. (i+1)*dim].
/// @param query Query vector (dim elements)
/// @param vectors Packed database vectors (n * dim elements)
/// @param n Number of database vectors
/// @param dim Vector dimension
/// @param distances Output array (n elements, caller-allocated)
void zvec_compute_l2_distance_batch(
    const float* query,
    const float* vectors,
    size_t n,
    size_t dim,
    float* distances
);

#ifdef __cplusplus
}
#endif

#endif // ZVEC_C_API_H

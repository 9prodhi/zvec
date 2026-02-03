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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error Codes
// ============================================================================

#define ZVEC_OK 0
#define ZVEC_ERR_NOT_FOUND 1
#define ZVEC_ERR_ALREADY_EXISTS 2
#define ZVEC_ERR_INVALID_ARGUMENT 3
#define ZVEC_ERR_IO 4
#define ZVEC_ERR_CORRUPTION 5
#define ZVEC_ERR_NOT_SUPPORTED 6
#define ZVEC_ERR_INTERNAL 7
#define ZVEC_ERR_UNKNOWN 10

// ============================================================================
// Basic Types
// ============================================================================

/// A non-owning view into a string (borrowed)
typedef struct {
    const char* data;
    size_t len;
} ZvecStrView;

/// An owning string (must be freed with zvec_str_free)
typedef struct {
    char* data;
    size_t len;
} ZvecStr;

/// A non-owning view into an array (borrowed)
typedef struct {
    const void* data;
    size_t len;
} ZvecArrayView;

/// Status returned by operations
typedef struct {
    int32_t code;
    const char* msg;  // Points to thread-local storage, valid until next call
} ZvecStatus;

/// Collection options
typedef struct {
    bool read_only;
    bool enable_mmap;
    uint32_t max_buffer_size;
} ZvecCollectionOptions;

// ============================================================================
// Opaque Handle Types
// ============================================================================

typedef struct ZvecCollection ZvecCollection;
typedef struct ZvecDoc ZvecDoc;
typedef struct ZvecDocList ZvecDocList;
typedef struct ZvecDocMap ZvecDocMap;
typedef struct ZvecStatusList ZvecStatusList;
typedef struct ZvecCollectionSchema ZvecCollectionSchema;
typedef struct ZvecFieldSchema ZvecFieldSchema;
typedef struct ZvecIndexParams ZvecIndexParams;
typedef struct ZvecQueryParams ZvecQueryParams;
typedef struct ZvecVectorQuery ZvecVectorQuery;
typedef struct ZvecGroupByVectorQuery ZvecGroupByVectorQuery;
typedef struct ZvecGroupResults ZvecGroupResults;
typedef struct ZvecGroupResult ZvecGroupResult;

// ============================================================================
// Error Handling
// ============================================================================

/// Get the last error message from thread-local storage.
/// The returned pointer is valid until the next API call on this thread.
const char* zvec_last_error_message(void);

/// Free an owned string returned by the API.
void zvec_str_free(ZvecStr str);

// ============================================================================
// Index Parameters
// ============================================================================

/// Create HNSW index parameters.
ZvecIndexParams* zvec_index_params_hnsw_create(
    uint32_t metric_type,
    int32_t m,
    int32_t ef_construction,
    uint32_t quantize_type
);

/// Create IVF index parameters.
ZvecIndexParams* zvec_index_params_ivf_create(
    uint32_t metric_type,
    int32_t n_list,
    int32_t n_iters,
    bool use_soar,
    uint32_t quantize_type
);

/// Create flat index parameters.
ZvecIndexParams* zvec_index_params_flat_create(
    uint32_t metric_type,
    uint32_t quantize_type
);

/// Create inverted index parameters.
ZvecIndexParams* zvec_index_params_invert_create(
    bool range_optimized,
    bool extended_wildcard
);

/// Free index parameters.
void zvec_index_params_free(ZvecIndexParams* params);

// ============================================================================
// Query Parameters
// ============================================================================

/// Create HNSW query parameters.
ZvecQueryParams* zvec_query_params_hnsw_create(
    int32_t ef,
    float radius,
    bool is_linear,
    bool is_using_refiner
);

/// Create IVF query parameters.
ZvecQueryParams* zvec_query_params_ivf_create(
    int32_t n_probes,
    bool is_using_refiner,
    float scale_factor
);

/// Create flat query parameters.
ZvecQueryParams* zvec_query_params_flat_create(
    bool is_using_refiner,
    float scale_factor
);

/// Free query parameters.
void zvec_query_params_free(ZvecQueryParams* params);

// ============================================================================
// Field Schema
// ============================================================================

/// Create a field schema.
ZvecFieldSchema* zvec_field_schema_create(
    ZvecStrView name,
    uint32_t data_type,
    bool nullable,
    uint32_t dimension
);

/// Set index parameters for a field schema.
ZvecStatus zvec_field_schema_set_index_params(
    ZvecFieldSchema* schema,
    const ZvecIndexParams* params
);

/// Free a field schema.
void zvec_field_schema_free(ZvecFieldSchema* schema);

// ============================================================================
// Collection Schema
// ============================================================================

/// Create a collection schema.
ZvecCollectionSchema* zvec_collection_schema_create(ZvecStrView name);

/// Add a field to the collection schema.
ZvecStatus zvec_collection_schema_add_field(
    ZvecCollectionSchema* schema,
    const ZvecFieldSchema* field
);

/// Set the maximum document count per segment.
ZvecStatus zvec_collection_schema_set_max_doc_count_per_segment(
    ZvecCollectionSchema* schema,
    uint64_t count
);

/// Free a collection schema.
void zvec_collection_schema_free(ZvecCollectionSchema* schema);

// ============================================================================
// Collection
// ============================================================================

/// Create and open a new collection.
ZvecStatus zvec_collection_create_and_open(
    ZvecStrView path,
    const ZvecCollectionSchema* schema,
    const ZvecCollectionOptions* options,
    ZvecCollection** out
);

/// Open an existing collection.
ZvecStatus zvec_collection_open(
    ZvecStrView path,
    const ZvecCollectionOptions* options,
    ZvecCollection** out
);

/// Close and free a collection.
void zvec_collection_free(ZvecCollection* coll);

/// Destroy a collection (permanently delete all data).
ZvecStatus zvec_collection_destroy(ZvecCollection* coll);

/// Get the collection path.
ZvecStatus zvec_collection_path(const ZvecCollection* coll, ZvecStr* out);

/// Get collection statistics as a string.
ZvecStatus zvec_collection_stats_string(const ZvecCollection* coll, ZvecStr* out);

/// Get collection schema as a string.
ZvecStatus zvec_collection_schema_string(const ZvecCollection* coll, ZvecStr* out);

/// Get collection options.
ZvecStatus zvec_collection_options(const ZvecCollection* coll, ZvecCollectionOptions* out);

/// Flush pending writes to disk.
ZvecStatus zvec_collection_flush(ZvecCollection* coll);

/// Optimize the collection.
ZvecStatus zvec_collection_optimize(ZvecCollection* coll, int32_t concurrency);

/// Create an index on a column.
ZvecStatus zvec_collection_create_index(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecIndexParams* params,
    int32_t concurrency
);

/// Drop an index from a column.
ZvecStatus zvec_collection_drop_index(ZvecCollection* coll, ZvecStrView column);

/// Add a column to the collection.
ZvecStatus zvec_collection_add_column(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecFieldSchema* field,
    ZvecStrView expression,
    int32_t concurrency
);

/// Drop a column from the collection.
ZvecStatus zvec_collection_drop_column(ZvecCollection* coll, ZvecStrView column);

/// Alter a column in the collection.
ZvecStatus zvec_collection_alter_column(
    ZvecCollection* coll,
    ZvecStrView column,
    ZvecStrView new_name,
    const ZvecFieldSchema* new_schema,  // May be NULL
    int32_t concurrency
);

/// Insert documents into the collection.
ZvecStatus zvec_collection_insert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Upsert documents into the collection.
ZvecStatus zvec_collection_upsert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Update documents in the collection.
ZvecStatus zvec_collection_update(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
);

/// Delete documents by primary key.
ZvecStatus zvec_collection_delete(
    ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecStatusList** out
);

/// Delete documents matching a filter.
ZvecStatus zvec_collection_delete_by_filter(ZvecCollection* coll, ZvecStrView filter);

/// Execute a vector query.
ZvecStatus zvec_collection_query(
    const ZvecCollection* coll,
    const ZvecVectorQuery* query,
    ZvecDocList** out
);

/// Execute a group-by vector query.
ZvecStatus zvec_collection_group_by_query(
    const ZvecCollection* coll,
    const ZvecGroupByVectorQuery* query,
    ZvecGroupResults** out
);

/// Fetch documents by primary key.
ZvecStatus zvec_collection_fetch(
    const ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecDocMap** out
);

// ============================================================================
// Document
// ============================================================================

/// Create a new empty document.
ZvecDoc* zvec_doc_create(void);

/// Free a document.
void zvec_doc_free(ZvecDoc* doc);

/// Set the primary key of a document.
void zvec_doc_set_pk(ZvecDoc* doc, ZvecStrView pk);

/// Set the operator of a document.
void zvec_doc_set_operator(ZvecDoc* doc, uint32_t op);

/// Set a boolean field.
ZvecStatus zvec_doc_set_bool(ZvecDoc* doc, ZvecStrView field, bool value);

/// Set a 32-bit integer field.
ZvecStatus zvec_doc_set_i32(ZvecDoc* doc, ZvecStrView field, int32_t value);

/// Set a 64-bit integer field.
ZvecStatus zvec_doc_set_i64(ZvecDoc* doc, ZvecStrView field, int64_t value);

/// Set a 32-bit unsigned integer field.
ZvecStatus zvec_doc_set_u32(ZvecDoc* doc, ZvecStrView field, uint32_t value);

/// Set a 64-bit unsigned integer field.
ZvecStatus zvec_doc_set_u64(ZvecDoc* doc, ZvecStrView field, uint64_t value);

/// Set a 32-bit float field.
ZvecStatus zvec_doc_set_f32(ZvecDoc* doc, ZvecStrView field, float value);

/// Set a 64-bit float field.
ZvecStatus zvec_doc_set_f64(ZvecDoc* doc, ZvecStrView field, double value);

/// Set a string field.
ZvecStatus zvec_doc_set_string(ZvecDoc* doc, ZvecStrView field, ZvecStrView value);

/// Set a dense 32-bit float vector field.
ZvecStatus zvec_doc_set_dense_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const float* data,
    size_t len
);

/// Set a dense 64-bit float vector field.
ZvecStatus zvec_doc_set_dense_f64(
    ZvecDoc* doc,
    ZvecStrView field,
    const double* data,
    size_t len
);

/// Set a sparse 32-bit float vector field.
ZvecStatus zvec_doc_set_sparse_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const uint32_t* indices,
    const float* values,
    size_t len
);

/// Get the primary key of a document (returns owned string).
ZvecStr zvec_doc_get_pk(const ZvecDoc* doc);

/// Get the score of a document.
float zvec_doc_get_score(const ZvecDoc* doc);

/// Get the doc_id of a document.
uint64_t zvec_doc_get_doc_id(const ZvecDoc* doc);

/// Check if a field exists in a document.
bool zvec_doc_has_field(const ZvecDoc* doc, ZvecStrView field);

/// Get a string field from a document.
ZvecStatus zvec_doc_get_string_field(const ZvecDoc* doc, ZvecStrView field, ZvecStr* out);

/// Get a dense vector field from a document.
ZvecStatus zvec_doc_get_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* out_data_type,
    ZvecArrayView* out_data
);

/// Get a sparse vector field from a document.
ZvecStatus zvec_doc_get_sparse_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* out_data_type,
    ZvecArrayView* out_indices,
    ZvecArrayView* out_values
);

/// Get the type of a field.
ZvecStatus zvec_doc_get_field_type(
    const ZvecDoc* doc,
    ZvecStrView field,
    bool* out_found,
    bool* out_is_null,
    uint32_t* out_data_type
);

/// Convert a document to a string representation.
ZvecStr zvec_doc_to_string(const ZvecDoc* doc);

/// Convert a document to a detailed string representation.
ZvecStr zvec_doc_to_detail_string(const ZvecDoc* doc);

// ============================================================================
// Document List
// ============================================================================

/// Get the length of a document list.
size_t zvec_doc_list_len(const ZvecDocList* list);

/// Get a document from the list by index (borrowed reference).
const ZvecDoc* zvec_doc_list_get(const ZvecDocList* list, size_t index);

/// Free a document list.
void zvec_doc_list_free(ZvecDocList* list);

// ============================================================================
// Document Map
// ============================================================================

/// Get the length of a document map.
size_t zvec_doc_map_len(const ZvecDocMap* map);

/// Get a key from the map by index (returns owned string).
ZvecStr zvec_doc_map_get_key(const ZvecDocMap* map, size_t index);

/// Get a value from the map by index (borrowed reference).
const ZvecDoc* zvec_doc_map_get_value(const ZvecDocMap* map, size_t index);

/// Free a document map.
void zvec_doc_map_free(ZvecDocMap* map);

// ============================================================================
// Status List
// ============================================================================

/// Get the length of a status list.
size_t zvec_status_list_len(const ZvecStatusList* list);

/// Get a status from the list by index.
ZvecStatus zvec_status_list_get(const ZvecStatusList* list, size_t index);

/// Free a status list.
void zvec_status_list_free(ZvecStatusList* list);

// ============================================================================
// Vector Query
// ============================================================================

/// Create a vector query.
ZvecVectorQuery* zvec_vector_query_create(
    ZvecStrView field_name,
    const void* dense_data,
    size_t dense_len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    int32_t topk
);

/// Set a filter expression on the query.
void zvec_vector_query_set_filter(ZvecVectorQuery* query, ZvecStrView filter);

/// Set output fields for the query.
void zvec_vector_query_set_output_fields(
    ZvecVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
);

/// Set query parameters.
void zvec_vector_query_set_query_params(ZvecVectorQuery* query, const ZvecQueryParams* params);

/// Set whether to include the vector in results.
void zvec_vector_query_set_include_vector(ZvecVectorQuery* query, bool include);

/// Set whether to include the doc_id in results.
void zvec_vector_query_set_include_doc_id(ZvecVectorQuery* query, bool include);

/// Free a vector query.
void zvec_vector_query_free(ZvecVectorQuery* query);

// ============================================================================
// Group By Vector Query
// ============================================================================

/// Create a group-by vector query.
ZvecGroupByVectorQuery* zvec_group_by_vector_query_create(
    ZvecStrView field_name,
    const void* dense_data,
    size_t dense_len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    ZvecStrView group_by_field,
    uint32_t group_count,
    uint32_t group_topk
);

/// Set a filter expression on the query.
void zvec_group_by_vector_query_set_filter(ZvecGroupByVectorQuery* query, ZvecStrView filter);

/// Set output fields for the query.
void zvec_group_by_vector_query_set_output_fields(
    ZvecGroupByVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
);

/// Set query parameters.
void zvec_group_by_vector_query_set_query_params(
    ZvecGroupByVectorQuery* query,
    const ZvecQueryParams* params
);

/// Set whether to include the vector in results.
void zvec_group_by_vector_query_set_include_vector(ZvecGroupByVectorQuery* query, bool include);

/// Free a group-by vector query.
void zvec_group_by_vector_query_free(ZvecGroupByVectorQuery* query);

// ============================================================================
// Group Results
// ============================================================================

/// Get the number of groups.
size_t zvec_group_results_len(const ZvecGroupResults* results);

/// Get a group result by index.
const ZvecGroupResult* zvec_group_results_get(const ZvecGroupResults* results, size_t index);

/// Free group results.
void zvec_group_results_free(ZvecGroupResults* results);

/// Get the group value from a group result.
ZvecStr zvec_group_result_get_group_value(const ZvecGroupResult* result);

/// Get the document list from a group result.
const ZvecDocList* zvec_group_result_get_docs(const ZvecGroupResult* result);

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_C_API_H

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

#include "zvec_c_api.h"

#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/index_params.h>
#include <zvec/db/query_params.h>
#include <zvec/db/options.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

// ============================================================================
// Internal types and helpers
// ============================================================================

// Thread-local error storage
thread_local std::string g_last_error;

// Error code for internal/unknown errors
static const int32_t ZVEC_ERR_INTERNAL = 7;

// Exception handling macros for FFI safety
#define CATCH_BEGIN try {
#define CATCH_END                                                     \
    }                                                                 \
    catch (const std::exception& e) {                                 \
        g_last_error = e.what();                                      \
        return ZvecStatus{ZVEC_ERR_INTERNAL, g_last_error.c_str()};   \
    }                                                                 \
    catch (...) {                                                     \
        g_last_error = "unknown C++ exception";                       \
        return ZvecStatus{ZVEC_ERR_INTERNAL, g_last_error.c_str()};   \
    }

#define CATCH_END_VOID                                                \
    }                                                                 \
    catch (const std::exception& e) {                                 \
        g_last_error = e.what();                                      \
    }                                                                 \
    catch (...) {                                                     \
        g_last_error = "unknown C++ exception";                       \
    }

// Wrapper structs for opaque handles
struct ZvecCollection {
    zvec::Collection::Ptr ptr;
};

struct ZvecDoc {
    std::shared_ptr<zvec::Doc> ptr;
};

struct ZvecDocList {
    zvec::DocPtrList list;
    std::vector<std::string> keys;  // Cache for iteration
};

struct ZvecDocMap {
    zvec::DocPtrMap map;
    std::vector<std::string> keys;  // Cache for iteration
};

struct ZvecFieldSchema {
    std::shared_ptr<zvec::FieldSchema> ptr;
};

struct ZvecCollectionSchema {
    std::shared_ptr<zvec::CollectionSchema> ptr;
};

struct ZvecIndexParams {
    zvec::IndexParams::Ptr ptr;
};

struct ZvecQueryParams {
    zvec::QueryParams::Ptr ptr;
};

struct ZvecVectorQuery {
    std::unique_ptr<zvec::VectorQuery> query;
};

struct ZvecGroupByVectorQuery {
    std::unique_ptr<zvec::GroupByVectorQuery> query;
};

struct ZvecStatusList {
    std::vector<zvec::Status> statuses;
};

struct ZvecGroupResults {
    std::vector<zvec::GroupResult> results;
};

// Note: ZvecGroupResult is not allocated separately, it's accessed from ZvecGroupResults

// Helper: convert ZvecStrView to std::string
static std::string to_string(ZvecStrView v) {
    return std::string(v.data, v.len);
}

// Helper: convert ZvecStrView to std::string_view
// Note: Currently unused but kept for potential future use
[[maybe_unused]] static std::string_view to_sv(ZvecStrView v) {
    return std::string_view(v.data, v.len);
}

// Helper: create ZvecStr from std::string (allocates)
static ZvecStr make_str(const std::string& s) {
    ZvecStr result;
    result.len = s.size();
    result.data = static_cast<char*>(malloc(s.size() + 1));
    if (result.data) {
        memcpy(result.data, s.data(), s.size());
        result.data[s.size()] = '\0';
    }
    return result;
}

// Helper: create success status
static ZvecStatus make_ok() {
    return {0, nullptr};
}

// Helper: create error status from zvec::Status
static ZvecStatus make_status(const zvec::Status& s) {
    if (s.ok()) {
        return make_ok();
    }
    g_last_error = s.message();
    return {static_cast<int32_t>(s.code()), g_last_error.c_str()};
}

// Helper: create error status with message
static ZvecStatus make_error(int32_t code, const std::string& msg) {
    g_last_error = msg;
    return {code, g_last_error.c_str()};
}

// Helper: convert C enum to zvec enum
static zvec::DataType to_data_type(uint32_t t) {
    return static_cast<zvec::DataType>(t);
}

static zvec::MetricType to_metric_type(uint32_t t) {
    return static_cast<zvec::MetricType>(t);
}

static zvec::QuantizeType to_quantize_type(uint32_t t) {
    return static_cast<zvec::QuantizeType>(t);
}

// ============================================================================
// Error handling
// ============================================================================

extern "C" {

const char* zvec_last_error_message(void) {
    return g_last_error.empty() ? "unknown error" : g_last_error.c_str();
}

void zvec_str_free(ZvecStr s) {
    if (s.data) {
        free(s.data);
    }
}

// ============================================================================
// Field Schema APIs
// ============================================================================

ZvecFieldSchema* zvec_field_schema_create(
    ZvecStrView name,
    uint32_t data_type,
    bool nullable,
    uint32_t dimension
) {
    auto fs = new ZvecFieldSchema();
    fs->ptr = std::make_shared<zvec::FieldSchema>(
        to_string(name),
        to_data_type(data_type),
        dimension,
        nullable,
        nullptr
    );
    return fs;
}

void zvec_field_schema_free(ZvecFieldSchema* fs) {
    delete fs;
}

ZvecStatus zvec_field_schema_set_index_params(
    ZvecFieldSchema* fs,
    const ZvecIndexParams* params
) {
    if (!fs || !fs->ptr) {
        return make_error(-1, "null field schema");
    }
    if (params && params->ptr) {
        fs->ptr->set_index_params(params->ptr);
    }
    return make_ok();
}

// ============================================================================
// Collection Schema APIs
// ============================================================================

ZvecCollectionSchema* zvec_collection_schema_create(ZvecStrView name) {
    auto cs = new ZvecCollectionSchema();
    cs->ptr = std::make_shared<zvec::CollectionSchema>(to_string(name));
    return cs;
}

void zvec_collection_schema_free(ZvecCollectionSchema* cs) {
    delete cs;
}

ZvecStatus zvec_collection_schema_add_field(
    ZvecCollectionSchema* cs,
    const ZvecFieldSchema* fs
) {
    if (!cs || !cs->ptr) {
        return make_error(-1, "null collection schema");
    }
    if (!fs || !fs->ptr) {
        return make_error(-1, "null field schema");
    }
    auto status = cs->ptr->add_field(fs->ptr);
    return make_status(status);
}

ZvecStatus zvec_collection_schema_set_max_doc_count_per_segment(
    ZvecCollectionSchema* cs,
    uint64_t count
) {
    if (!cs || !cs->ptr) {
        return make_error(-1, "null collection schema");
    }
    cs->ptr->set_max_doc_count_per_segment(count);
    return make_ok();
}

// ============================================================================
// Index Parameters APIs
// ============================================================================

ZvecIndexParams* zvec_index_params_hnsw_create(
    uint32_t metric,
    int32_t m,
    int32_t ef_construction,
    uint32_t quantize
) {
    auto params = new ZvecIndexParams();
    params->ptr = std::make_shared<zvec::HnswIndexParams>(
        to_metric_type(metric),
        m,
        ef_construction,
        to_quantize_type(quantize)
    );
    return params;
}

ZvecIndexParams* zvec_index_params_ivf_create(
    uint32_t metric,
    int32_t n_list,
    int32_t n_iters,
    bool use_soar,
    uint32_t quantize
) {
    auto params = new ZvecIndexParams();
    params->ptr = std::make_shared<zvec::IVFIndexParams>(
        to_metric_type(metric),
        n_list,
        n_iters,
        use_soar,
        to_quantize_type(quantize)
    );
    return params;
}

ZvecIndexParams* zvec_index_params_flat_create(
    uint32_t metric,
    uint32_t quantize
) {
    auto params = new ZvecIndexParams();
    params->ptr = std::make_shared<zvec::FlatIndexParams>(
        to_metric_type(metric),
        to_quantize_type(quantize)
    );
    return params;
}

ZvecIndexParams* zvec_index_params_invert_create(
    bool range_optimized,
    bool extended_wildcard
) {
    auto params = new ZvecIndexParams();
    params->ptr = std::make_shared<zvec::InvertIndexParams>(
        range_optimized,
        extended_wildcard
    );
    return params;
}

void zvec_index_params_free(ZvecIndexParams* params) {
    delete params;
}

// ============================================================================
// Query Parameters APIs
// ============================================================================

ZvecQueryParams* zvec_query_params_hnsw_create(
    int32_t ef,
    float radius,
    bool is_linear,
    bool is_using_refiner
) {
    auto params = new ZvecQueryParams();
    // HnswQueryParams constructor: (ef, radius, is_linear, is_using_refiner)
    params->ptr = std::make_shared<zvec::HnswQueryParams>(ef, radius, is_linear, is_using_refiner);
    return params;
}

ZvecQueryParams* zvec_query_params_ivf_create(
    int32_t n_probes,
    bool is_using_refiner,
    float scale_factor
) {
    auto params = new ZvecQueryParams();
    // IVFQueryParams constructor: (nprobe, is_using_refiner, scale_factor)
    params->ptr = std::make_shared<zvec::IVFQueryParams>(n_probes, is_using_refiner, scale_factor);
    return params;
}

ZvecQueryParams* zvec_query_params_flat_create(
    bool is_using_refiner,
    float scale_factor
) {
    auto params = new ZvecQueryParams();
    // FlatQueryParams constructor: (is_using_refiner, scale_factor)
    params->ptr = std::make_shared<zvec::FlatQueryParams>(is_using_refiner, scale_factor);
    return params;
}

void zvec_query_params_free(ZvecQueryParams* params) {
    delete params;
}

// ============================================================================
// Collection Lifecycle APIs
// ============================================================================

ZvecStatus zvec_collection_create_and_open(
    ZvecStrView path,
    const ZvecCollectionSchema* schema,
    const ZvecCollectionOptions* opts,
    ZvecCollection** out
) {
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    if (!schema || !schema->ptr) {
        return make_error(-1, "null schema");
    }
    CATCH_BEGIN

    zvec::CollectionOptions options;
    if (opts) {
        options.read_only_ = opts->read_only;
        options.enable_mmap_ = opts->enable_mmap;
        options.max_buffer_size_ = opts->max_buffer_size;
    }

    auto result = zvec::Collection::CreateAndOpen(
        to_string(path),
        *schema->ptr,
        options
    );

    if (!result) {
        return make_status(result.error());
    }

    auto coll = new ZvecCollection();
    coll->ptr = std::move(result.value());
    *out = coll;
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_open(
    ZvecStrView path,
    const ZvecCollectionOptions* opts,
    ZvecCollection** out
) {
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN

    zvec::CollectionOptions options;
    if (opts) {
        options.read_only_ = opts->read_only;
        options.enable_mmap_ = opts->enable_mmap;
        options.max_buffer_size_ = opts->max_buffer_size;
    }

    auto result = zvec::Collection::Open(to_string(path), options);

    if (!result) {
        return make_status(result.error());
    }

    auto coll = new ZvecCollection();
    coll->ptr = std::move(result.value());
    *out = coll;
    return make_ok();
    CATCH_END
}

void zvec_collection_free(ZvecCollection* coll) {
    delete coll;
}

ZvecStatus zvec_collection_destroy(ZvecCollection* coll) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    auto status = coll->ptr->Destroy();
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_flush(ZvecCollection* coll) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    auto status = coll->ptr->Flush();
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_optimize(ZvecCollection* coll, int32_t concurrency) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    zvec::OptimizeOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->Optimize(opts);
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_path(const ZvecCollection* coll, ZvecStr* out) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN
    auto result = coll->ptr->Path();
    if (!result) {
        return make_status(result.error());
    }
    *out = make_str(result.value());
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_stats_string(const ZvecCollection* coll, ZvecStr* out) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN
    auto result = coll->ptr->Stats();
    if (!result) {
        return make_status(result.error());
    }
    // Use the to_string method from CollectionStats
    *out = make_str(result.value().to_string());
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_schema_string(const ZvecCollection* coll, ZvecStr* out) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN
    auto result = coll->ptr->Schema();
    if (!result) {
        return make_status(result.error());
    }
    // Convert schema to string
    *out = make_str(result.value().name());
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_options(const ZvecCollection* coll, ZvecCollectionOptions* out) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN
    auto result = coll->ptr->Options();
    if (!result) {
        return make_status(result.error());
    }
    auto opts = result.value();
    out->read_only = opts.read_only_;
    out->enable_mmap = opts.enable_mmap_;
    out->max_buffer_size = opts.max_buffer_size_;
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_create_index(
    ZvecCollection* coll,
    ZvecStrView field_name,
    const ZvecIndexParams* params,
    int32_t concurrency
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!params || !params->ptr) {
        return make_error(-1, "null index params");
    }
    CATCH_BEGIN
    zvec::CreateIndexOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->CreateIndex(to_string(field_name), params->ptr, opts);
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_drop_index(
    ZvecCollection* coll,
    ZvecStrView field_name
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    auto status = coll->ptr->DropIndex(to_string(field_name));
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_add_column(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecFieldSchema* field,
    ZvecStrView expression,
    int32_t concurrency
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!field || !field->ptr) {
        return make_error(-1, "null field schema");
    }
    CATCH_BEGIN
    zvec::AddColumnOptions opts;
    opts.concurrency_ = concurrency;
    // AddColumn(column_name, column_schema, expression, options)
    auto status = coll->ptr->AddColumn(to_string(column), field->ptr, to_string(expression), opts);
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_drop_column(
    ZvecCollection* coll,
    ZvecStrView column
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    auto status = coll->ptr->DropColumn(to_string(column));
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_alter_column(
    ZvecCollection* coll,
    ZvecStrView column,
    ZvecStrView new_name,
    const ZvecFieldSchema* new_schema,
    int32_t concurrency
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    zvec::AlterColumnOptions opts;
    opts.concurrency_ = concurrency;
    // AlterColumn(column_name, rename, new_column_schema, options)
    zvec::FieldSchema::Ptr schema_ptr = nullptr;
    if (new_schema && new_schema->ptr) {
        schema_ptr = new_schema->ptr;
    }
    auto status = coll->ptr->AlterColumn(to_string(column), to_string(new_name), schema_ptr, opts);
    return make_status(status);
    CATCH_END
}

// ============================================================================
// Document APIs
// ============================================================================

ZvecDoc* zvec_doc_create(void) {
    auto doc = new ZvecDoc();
    doc->ptr = std::make_shared<zvec::Doc>();
    return doc;
}

void zvec_doc_free(ZvecDoc* doc) {
    delete doc;
}

void zvec_doc_set_pk(ZvecDoc* doc, ZvecStrView pk) {
    if (doc && doc->ptr) {
        doc->ptr->set_pk(to_string(pk));
    }
}

ZvecStr zvec_doc_get_pk(const ZvecDoc* doc) {
    if (!doc || !doc->ptr) {
        return {nullptr, 0};
    }
    return make_str(doc->ptr->pk());
}

float zvec_doc_get_score(const ZvecDoc* doc) {
    if (!doc || !doc->ptr) {
        return 0.0f;
    }
    return doc->ptr->score();
}

uint64_t zvec_doc_get_doc_id(const ZvecDoc* doc) {
    if (!doc || !doc->ptr) {
        return 0;
    }
    return doc->ptr->doc_id();
}

void zvec_doc_set_operator(ZvecDoc* doc, uint32_t op) {
    if (doc && doc->ptr) {
        doc->ptr->set_operator(static_cast<zvec::Operator>(op));
    }
}

bool zvec_doc_has_field(const ZvecDoc* doc, ZvecStrView field) {
    if (!doc || !doc->ptr) {
        return false;
    }
    return doc->ptr->has(to_string(field));
}

ZvecStatus zvec_doc_set_bool(ZvecDoc* doc, ZvecStrView field, bool value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_i32(ZvecDoc* doc, ZvecStrView field, int32_t value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_i64(ZvecDoc* doc, ZvecStrView field, int64_t value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_u32(ZvecDoc* doc, ZvecStrView field, uint32_t value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_u64(ZvecDoc* doc, ZvecStrView field, uint64_t value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_f32(ZvecDoc* doc, ZvecStrView field, float value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_f64(ZvecDoc* doc, ZvecStrView field, double value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), value);
    return make_ok();
}

ZvecStatus zvec_doc_set_string(ZvecDoc* doc, ZvecStrView field, ZvecStrView value) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    doc->ptr->set(to_string(field), to_string(value));
    return make_ok();
}

ZvecStatus zvec_doc_set_dense_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const float* data,
    size_t len
) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    std::vector<float> vec(data, data + len);
    doc->ptr->set(to_string(field), std::move(vec));
    return make_ok();
}

ZvecStatus zvec_doc_set_dense_f64(
    ZvecDoc* doc,
    ZvecStrView field,
    const double* data,
    size_t len
) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    std::vector<double> vec(data, data + len);
    doc->ptr->set(to_string(field), std::move(vec));
    return make_ok();
}

ZvecStatus zvec_doc_set_sparse_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const uint32_t* indices,
    const float* values,
    size_t len
) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    std::vector<uint32_t> idx(indices, indices + len);
    std::vector<float> vals(values, values + len);
    doc->ptr->set(to_string(field), std::make_pair(std::move(idx), std::move(vals)));
    return make_ok();
}

ZvecStatus zvec_doc_get_string(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecStr* out
) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN
    auto result = doc->ptr->get<std::string>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = make_str(*result);
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_string_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecStr* out
) {
    // Alias for zvec_doc_get_string
    return zvec_doc_get_string(doc, field, out);
}

ZvecStatus zvec_doc_get_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* data_type,
    ZvecArrayView* out
) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN

    static thread_local std::vector<float> cached_f32;
    static thread_local std::vector<double> cached_f64;

    auto result_f32 = doc->ptr->get<std::vector<float>>(to_string(field));
    if (result_f32.has_value()) {
        cached_f32 = std::move(*result_f32);
        if (data_type) *data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP32);
        out->data = cached_f32.data();
        out->len = cached_f32.size();
        return make_ok();
    }

    auto result_f64 = doc->ptr->get<std::vector<double>>(to_string(field));
    if (result_f64.has_value()) {
        cached_f64 = std::move(*result_f64);
        if (data_type) *data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP64);
        out->data = cached_f64.data();
        out->len = cached_f64.size();
        return make_ok();
    }

    return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    CATCH_END
}

ZvecStatus zvec_doc_get_sparse_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* data_type,
    ZvecArrayView* indices_out,
    ZvecArrayView* values_out
) {
    if (!doc || !doc->ptr || !indices_out || !values_out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN

    static thread_local std::pair<std::vector<uint32_t>, std::vector<float>> cached_sparse;

    auto result = doc->ptr->get<std::pair<std::vector<uint32_t>, std::vector<float>>>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }

    cached_sparse = std::move(*result);
    if (data_type) *data_type = static_cast<uint32_t>(zvec::DataType::SPARSE_VECTOR_FP32);
    indices_out->data = cached_sparse.first.data();
    indices_out->len = cached_sparse.first.size();
    values_out->data = cached_sparse.second.data();
    values_out->len = cached_sparse.second.size();
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_field_type(
    const ZvecDoc* doc,
    ZvecStrView field,
    bool* found,
    bool* is_null,
    uint32_t* data_type
) {
    if (!doc || !doc->ptr) {
        return make_error(-1, "null document");
    }
    CATCH_BEGIN

    std::string field_name = to_string(field);
    bool has_field = doc->ptr->has(field_name);
    if (found) *found = has_field;

    if (!has_field) {
        if (is_null) *is_null = false;
        if (data_type) *data_type = 0;  // Unknown
        return make_ok();
    }

    bool field_is_null = doc->ptr->is_null(field_name);
    if (is_null) *is_null = field_is_null;

    // Try to determine the data type by attempting to get different types
    // This is not ideal but Doc doesn't expose direct type information
    if (data_type) {
        if (field_is_null) {
            *data_type = 0;  // Null/unknown
        } else if (doc->ptr->get<std::vector<float>>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP32);
        } else if (doc->ptr->get<std::vector<double>>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP64);
        } else if (doc->ptr->get<std::string>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::STRING);
        } else if (doc->ptr->get<int64_t>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::INT64);
        } else if (doc->ptr->get<int32_t>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::INT32);
        } else if (doc->ptr->get<double>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::DOUBLE);
        } else if (doc->ptr->get<float>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::FLOAT);
        } else if (doc->ptr->get<bool>(field_name).has_value()) {
            *data_type = static_cast<uint32_t>(zvec::DataType::BOOL);
        } else {
            *data_type = 0;  // Unknown type
        }
    }
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_dense_f32(
    const ZvecDoc* doc,
    ZvecStrView field,
    ZvecArrayView* out
) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN

    static thread_local std::vector<float> cached_vector;

    auto result = doc->ptr->get<std::vector<float>>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }

    cached_vector = std::move(*result);
    out->data = cached_vector.data();
    out->len = cached_vector.size();
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_i64(const ZvecDoc* doc, ZvecStrView field, int64_t* out) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN
    auto result = doc->ptr->get<int64_t>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = *result;
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_f64(const ZvecDoc* doc, ZvecStrView field, double* out) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN
    auto result = doc->ptr->get<double>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = *result;
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_get_bool(const ZvecDoc* doc, ZvecStrView field, bool* out) {
    if (!doc || !doc->ptr || !out) {
        return make_error(-1, "null argument");
    }
    CATCH_BEGIN
    auto result = doc->ptr->get<bool>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = *result;
    return make_ok();
    CATCH_END
}

ZvecStr zvec_doc_to_string(const ZvecDoc* doc) {
    if (!doc || !doc->ptr) {
        return {nullptr, 0};
    }
    return make_str(doc->ptr->to_string());
}

ZvecStr zvec_doc_to_detail_string(const ZvecDoc* doc) {
    if (!doc || !doc->ptr) {
        return {nullptr, 0};
    }
    return make_str(doc->ptr->to_detail_string());
}

// ============================================================================
// Batch Insert/Update APIs
// ============================================================================

ZvecStatus zvec_collection_insert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!docs || count == 0) {
        return make_error(-1, "empty documents");
    }
    CATCH_BEGIN

    std::vector<zvec::Doc> zvec_docs;
    zvec_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            zvec_docs.push_back(*docs[i]->ptr);
        }
    }

    auto result = coll->ptr->Insert(zvec_docs);
    if (!result) {
        return make_status(result.error());
    }

    if (out) {
        auto status_list = new ZvecStatusList();
        status_list->statuses = std::move(result.value());
        *out = status_list;
    }
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_upsert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!docs || count == 0) {
        return make_error(-1, "empty documents");
    }
    CATCH_BEGIN

    std::vector<zvec::Doc> zvec_docs;
    zvec_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            zvec_docs.push_back(*docs[i]->ptr);
        }
    }

    auto result = coll->ptr->Upsert(zvec_docs);
    if (!result) {
        return make_status(result.error());
    }

    if (out) {
        auto status_list = new ZvecStatusList();
        status_list->statuses = std::move(result.value());
        *out = status_list;
    }
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_update(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!docs || count == 0) {
        return make_error(-1, "empty documents");
    }
    CATCH_BEGIN

    std::vector<zvec::Doc> zvec_docs;
    zvec_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            zvec_docs.push_back(*docs[i]->ptr);
        }
    }

    auto result = coll->ptr->Update(zvec_docs);
    if (!result) {
        return make_status(result.error());
    }

    if (out) {
        auto status_list = new ZvecStatusList();
        status_list->statuses = std::move(result.value());
        *out = status_list;
    }
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_delete(
    ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecStatusList** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN

    std::vector<std::string> pk_vec;
    pk_vec.reserve(count);
    for (size_t i = 0; i < count; i++) {
        pk_vec.push_back(to_string(pks[i]));
    }

    auto result = coll->ptr->Delete(pk_vec);
    if (!result) {
        return make_status(result.error());
    }

    if (out) {
        auto status_list = new ZvecStatusList();
        status_list->statuses = std::move(result.value());
        *out = status_list;
    }
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_collection_delete_by_filter(
    ZvecCollection* coll,
    ZvecStrView filter
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    CATCH_BEGIN
    auto status = coll->ptr->DeleteByFilter(to_string(filter));
    return make_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_fetch(
    const ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecDocMap** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN

    std::vector<std::string> pk_vec;
    pk_vec.reserve(count);
    for (size_t i = 0; i < count; i++) {
        pk_vec.push_back(to_string(pks[i]));
    }

    auto result = coll->ptr->Fetch(pk_vec);
    if (!result) {
        return make_status(result.error());
    }

    auto doc_map = new ZvecDocMap();
    doc_map->map = std::move(result.value());
    // Build keys cache for iteration
    for (const auto& [k, v] : doc_map->map) {
        doc_map->keys.push_back(k);
    }
    *out = doc_map;
    return make_ok();
    CATCH_END
}

// ============================================================================
// Vector Query APIs
// ============================================================================

ZvecVectorQuery* zvec_vector_query_create(
    ZvecStrView field,
    const void* data,
    size_t len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    int32_t topk
) {
    auto query = new ZvecVectorQuery();
    query->query = std::make_unique<zvec::VectorQuery>();
    query->query->field_name_ = to_string(field);
    query->query->topk_ = topk;

    // Copy dense vector data (len is in bytes)
    if (data && len > 0) {
        const char* raw_data = static_cast<const char*>(data);
        query->query->query_vector_ = std::string(raw_data, len);
    }

    // Copy sparse vector data if provided - stored as raw string bytes
    if (sparse_indices && sparse_values && sparse_len > 0) {
        // Indices stored as raw bytes
        const char* indices_ptr = reinterpret_cast<const char*>(sparse_indices);
        query->query->query_sparse_indices_ = std::string(indices_ptr, sparse_len * sizeof(uint32_t));
        // Values stored as raw bytes
        const char* values_ptr = static_cast<const char*>(sparse_values);
        query->query->query_sparse_values_ = std::string(values_ptr, sparse_len * sizeof(float));
    }

    return query;
}

void zvec_vector_query_free(ZvecVectorQuery* query) {
    delete query;
}

void zvec_vector_query_set_filter(ZvecVectorQuery* query, ZvecStrView filter) {
    if (query && query->query) {
        query->query->filter_ = to_string(filter);
    }
}

void zvec_vector_query_set_query_params(
    ZvecVectorQuery* query,
    const ZvecQueryParams* params
) {
    if (query && query->query && params && params->ptr) {
        query->query->query_params_ = params->ptr;
    }
}

void zvec_vector_query_set_include_vector(ZvecVectorQuery* query, bool include) {
    if (query && query->query) {
        query->query->include_vector_ = include;
    }
}

void zvec_vector_query_set_include_doc_id(ZvecVectorQuery* query, bool include) {
    if (query && query->query) {
        query->query->include_doc_id_ = include;
    }
}

void zvec_vector_query_set_output_fields(
    ZvecVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
) {
    if (query && query->query) {
        if (fields && count > 0) {
            std::vector<std::string> output_fields;
            output_fields.reserve(count);
            for (size_t i = 0; i < count; i++) {
                output_fields.push_back(to_string(fields[i]));
            }
            query->query->output_fields_ = std::move(output_fields);
        } else {
            query->query->output_fields_ = std::nullopt;
        }
    }
}

ZvecStatus zvec_collection_query(
    const ZvecCollection* coll,
    const ZvecVectorQuery* query,
    ZvecDocList** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!query || !query->query) {
        return make_error(-1, "null query");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN

    auto result = coll->ptr->Query(*query->query);
    if (!result) {
        return make_status(result.error());
    }

    auto doc_list = new ZvecDocList();
    doc_list->list = std::move(result.value());
    *out = doc_list;
    return make_ok();
    CATCH_END
}

// ============================================================================
// Group By Vector Query APIs
// ============================================================================

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
) {
    auto query = new ZvecGroupByVectorQuery();
    query->query = std::make_unique<zvec::GroupByVectorQuery>();
    query->query->field_name_ = to_string(field);
    query->query->group_by_field_name_ = to_string(group_by_field);
    query->query->group_count_ = group_count;
    query->query->group_topk_ = group_topk;

    // Copy dense vector data (len is in bytes)
    if (data && len > 0) {
        const char* raw_data = static_cast<const char*>(data);
        query->query->query_vector_ = std::string(raw_data, len);
    }

    // Copy sparse vector data if provided - stored as raw string bytes
    if (sparse_indices && sparse_values && sparse_len > 0) {
        // Indices stored as raw bytes
        const char* indices_ptr = reinterpret_cast<const char*>(sparse_indices);
        query->query->query_sparse_indices_ = std::string(indices_ptr, sparse_len * sizeof(uint32_t));
        // Values stored as raw bytes
        const char* values_ptr = static_cast<const char*>(sparse_values);
        query->query->query_sparse_values_ = std::string(values_ptr, sparse_len * sizeof(float));
    }

    return query;
}

void zvec_group_by_vector_query_free(ZvecGroupByVectorQuery* query) {
    delete query;
}

void zvec_group_by_vector_query_set_filter(ZvecGroupByVectorQuery* query, ZvecStrView filter) {
    if (query && query->query) {
        query->query->filter_ = to_string(filter);
    }
}

void zvec_group_by_vector_query_set_output_fields(
    ZvecGroupByVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
) {
    if (query && query->query) {
        if (fields && count > 0) {
            std::vector<std::string> output_fields;
            output_fields.reserve(count);
            for (size_t i = 0; i < count; i++) {
                output_fields.push_back(to_string(fields[i]));
            }
            query->query->output_fields_ = std::move(output_fields);
        } else {
            query->query->output_fields_ = std::nullopt;
        }
    }
}

void zvec_group_by_vector_query_set_query_params(
    ZvecGroupByVectorQuery* query,
    const ZvecQueryParams* params
) {
    if (query && query->query && params && params->ptr) {
        query->query->query_params_ = params->ptr;
    }
}

void zvec_group_by_vector_query_set_include_vector(ZvecGroupByVectorQuery* query, bool include) {
    if (query && query->query) {
        query->query->include_vector_ = include;
    }
}

ZvecStatus zvec_collection_group_by_query(
    const ZvecCollection* coll,
    const ZvecGroupByVectorQuery* query,
    ZvecGroupResults** out
) {
    if (!coll || !coll->ptr) {
        return make_error(-1, "null collection");
    }
    if (!query || !query->query) {
        return make_error(-1, "null query");
    }
    if (!out) {
        return make_error(-1, "null output pointer");
    }
    CATCH_BEGIN

    auto result = coll->ptr->GroupByQuery(*query->query);
    if (!result) {
        return make_status(result.error());
    }

    auto group_results = new ZvecGroupResults();
    group_results->results = std::move(result.value());
    *out = group_results;
    return make_ok();
    CATCH_END
}

// ============================================================================
// Result List Accessors (DocList)
// ============================================================================

size_t zvec_doc_list_len(const ZvecDocList* list) {
    return list ? list->list.size() : 0;
}

const ZvecDoc* zvec_doc_list_get(const ZvecDocList* list, size_t index) {
    if (!list || index >= list->list.size()) {
        return nullptr;
    }
    // Create a temporary wrapper - this is a bit awkward but safe
    static thread_local ZvecDoc temp_doc;
    temp_doc.ptr = list->list[index];
    return &temp_doc;
}

ZvecStr zvec_doc_list_get_pk(const ZvecDocList* list, size_t index) {
    if (!list || index >= list->list.size()) {
        return {nullptr, 0};
    }
    return make_str(list->list[index]->pk());
}

float zvec_doc_list_get_score(const ZvecDocList* list, size_t index) {
    if (!list || index >= list->list.size()) {
        return 0.0f;
    }
    return list->list[index]->score();
}

ZvecStatus zvec_doc_list_get_dense_f32(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    ZvecArrayView* out
) {
    if (!list || index >= list->list.size() || !out) {
        return make_error(-1, "invalid argument");
    }
    CATCH_BEGIN

    static thread_local std::vector<float> cached_vector;

    auto result = list->list[index]->get<std::vector<float>>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }

    cached_vector = std::move(*result);
    out->data = cached_vector.data();
    out->len = cached_vector.size();
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_list_get_string(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    ZvecStr* out
) {
    if (!list || index >= list->list.size() || !out) {
        return make_error(-1, "invalid argument");
    }
    CATCH_BEGIN
    auto result = list->list[index]->get<std::string>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = make_str(*result);
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_list_get_i64(
    const ZvecDocList* list,
    size_t index,
    ZvecStrView field,
    int64_t* out
) {
    if (!list || index >= list->list.size() || !out) {
        return make_error(-1, "invalid argument");
    }
    CATCH_BEGIN
    auto result = list->list[index]->get<int64_t>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = *result;
    return make_ok();
    CATCH_END
}

void zvec_doc_list_free(ZvecDocList* list) {
    delete list;
}

// ============================================================================
// Result Map Accessors (DocMap)
// ============================================================================

size_t zvec_doc_map_len(const ZvecDocMap* map) {
    return map ? map->map.size() : 0;
}

ZvecStr zvec_doc_map_get_key(const ZvecDocMap* map, size_t index) {
    if (!map || index >= map->keys.size()) {
        return {nullptr, 0};
    }
    return make_str(map->keys[index]);
}

const ZvecDoc* zvec_doc_map_get_value(const ZvecDocMap* map, size_t index) {
    if (!map || index >= map->keys.size()) {
        return nullptr;
    }
    const auto& key = map->keys[index];
    auto it = map->map.find(key);
    if (it == map->map.end()) {
        return nullptr;
    }
    // Create a temporary wrapper - this is a bit awkward but safe
    static thread_local ZvecDoc temp_doc;
    temp_doc.ptr = it->second;
    return &temp_doc;
}

const ZvecDoc* zvec_doc_map_get_doc(const ZvecDocMap* map, size_t index) {
    // Alias for zvec_doc_map_get_value
    return zvec_doc_map_get_value(map, index);
}

ZvecStatus zvec_doc_map_get_dense_f32(
    const ZvecDocMap* map,
    size_t index,
    ZvecStrView field,
    ZvecArrayView* out
) {
    if (!map || index >= map->keys.size() || !out) {
        return make_error(-1, "invalid argument");
    }
    CATCH_BEGIN

    const auto& key = map->keys[index];
    auto it = map->map.find(key);
    if (it == map->map.end()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "key not found");
    }

    static thread_local std::vector<float> cached_vector;

    auto result = it->second->get<std::vector<float>>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }

    cached_vector = std::move(*result);
    out->data = cached_vector.data();
    out->len = cached_vector.size();
    return make_ok();
    CATCH_END
}

ZvecStatus zvec_doc_map_get_string(
    const ZvecDocMap* map,
    size_t index,
    ZvecStrView field,
    ZvecStr* out
) {
    if (!map || index >= map->keys.size() || !out) {
        return make_error(-1, "invalid argument");
    }
    CATCH_BEGIN
    const auto& key = map->keys[index];
    auto it = map->map.find(key);
    if (it == map->map.end()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "key not found");
    }
    auto result = it->second->get<std::string>(to_string(field));
    if (!result.has_value()) {
        return make_error(static_cast<int32_t>(zvec::StatusCode::NOT_FOUND), "field not found");
    }
    *out = make_str(*result);
    return make_ok();
    CATCH_END
}

void zvec_doc_map_free(ZvecDocMap* map) {
    delete map;
}

// ============================================================================
// Status List Accessors
// ============================================================================

size_t zvec_status_list_len(const ZvecStatusList* list) {
    return list ? list->statuses.size() : 0;
}

ZvecStatus zvec_status_list_get(const ZvecStatusList* list, size_t index) {
    if (!list || index >= list->statuses.size()) {
        return make_error(-1, "invalid index");
    }
    return make_status(list->statuses[index]);
}

void zvec_status_list_free(ZvecStatusList* list) {
    delete list;
}

// ============================================================================
// Group Results Accessors
// ============================================================================

size_t zvec_group_results_len(const ZvecGroupResults* results) {
    return results ? results->results.size() : 0;
}

const ZvecGroupResult* zvec_group_results_get(const ZvecGroupResults* results, size_t index) {
    if (!results || index >= results->results.size()) {
        return nullptr;
    }
    // Return pointer to the result within the vector
    return reinterpret_cast<const ZvecGroupResult*>(&results->results[index]);
}

void zvec_group_results_free(ZvecGroupResults* results) {
    delete results;
}

ZvecStr zvec_group_result_get_group_value(const ZvecGroupResult* result) {
    if (!result) {
        return {nullptr, 0};
    }
    const auto* gr = reinterpret_cast<const zvec::GroupResult*>(result);
    return make_str(gr->group_by_value_);
}

const ZvecDocList* zvec_group_result_get_docs(const ZvecGroupResult* result) {
    if (!result) {
        return nullptr;
    }
    // GroupResult::docs_ is std::vector<Doc>, we need to convert to DocPtrList
    const auto* gr = reinterpret_cast<const zvec::GroupResult*>(result);
    static thread_local ZvecDocList temp_list;
    temp_list.list.clear();
    temp_list.list.reserve(gr->docs_.size());
    for (const auto& doc : gr->docs_) {
        temp_list.list.push_back(std::make_shared<zvec::Doc>(doc));
    }
    return &temp_list;
}

} // extern "C"

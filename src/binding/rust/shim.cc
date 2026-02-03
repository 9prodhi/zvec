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
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

// Thread-local error storage
thread_local std::string g_last_error;

// ============================================================================
// Internal Wrapper Types
// ============================================================================

struct ZvecCollection {
    zvec::Collection::Ptr ptr;
};

struct ZvecDoc {
    zvec::Doc doc;
};

struct ZvecDocList {
    zvec::DocPtrList docs;
};

struct ZvecDocMap {
    std::vector<std::pair<std::string, zvec::Doc::Ptr>> entries;
};

struct ZvecStatusList {
    std::vector<zvec::Status> statuses;
};

struct ZvecCollectionSchema {
    zvec::CollectionSchema schema;
};

struct ZvecFieldSchema {
    zvec::FieldSchema::Ptr ptr;
};

struct ZvecIndexParams {
    zvec::IndexParams::Ptr ptr;
};

struct ZvecQueryParams {
    zvec::QueryParams::Ptr ptr;
};

struct ZvecVectorQuery {
    zvec::VectorQuery query;
};

struct ZvecGroupByVectorQuery {
    zvec::GroupByVectorQuery query;
};

struct ZvecGroupResults {
    zvec::GroupResults results;
};

struct ZvecGroupResult {
    const zvec::GroupResult* result;
    ZvecDocList doc_list;
};

// ============================================================================
// Helper Macros and Functions
// ============================================================================

namespace {

inline std::string to_string(ZvecStrView view) {
    if (view.data == nullptr || view.len == 0) {
        return std::string();
    }
    return std::string(view.data, view.len);
}

inline ZvecStr to_zvec_str(const std::string& s) {
    ZvecStr result;
    result.len = s.size();
    result.data = static_cast<char*>(malloc(s.size() + 1));
    if (result.data) {
        memcpy(result.data, s.data(), s.size());
        result.data[s.size()] = '\0';
    }
    return result;
}

inline ZvecStatus make_status(int32_t code, const char* msg = nullptr) {
    if (msg) {
        g_last_error = msg;
    }
    return ZvecStatus{code, code != 0 ? g_last_error.c_str() : nullptr};
}

inline ZvecStatus from_status(const zvec::Status& status) {
    if (status.ok()) {
        return ZvecStatus{ZVEC_OK, nullptr};
    }
    g_last_error = status.message();
    int32_t code;
    switch (status.code()) {
        case zvec::StatusCode::NOT_FOUND:
            code = ZVEC_ERR_NOT_FOUND;
            break;
        case zvec::StatusCode::ALREADY_EXISTS:
            code = ZVEC_ERR_ALREADY_EXISTS;
            break;
        case zvec::StatusCode::INVALID_ARGUMENT:
            code = ZVEC_ERR_INVALID_ARGUMENT;
            break;
        case zvec::StatusCode::NOT_SUPPORTED:
            code = ZVEC_ERR_NOT_SUPPORTED;
            break;
        case zvec::StatusCode::INTERNAL_ERROR:
            code = ZVEC_ERR_INTERNAL;
            break;
        default:
            code = ZVEC_ERR_UNKNOWN;
            break;
    }
    return ZvecStatus{code, g_last_error.c_str()};
}

template <typename T>
inline ZvecStatus from_result(const zvec::Result<T>& result) {
    if (result.has_value()) {
        return ZvecStatus{ZVEC_OK, nullptr};
    }
    return from_status(result.error());
}

}  // namespace

#define CATCH_BEGIN try {
#define CATCH_END                                             \
    }                                                         \
    catch (const std::exception& e) {                         \
        g_last_error = e.what();                              \
        return ZvecStatus{ZVEC_ERR_INTERNAL, g_last_error.c_str()}; \
    }                                                         \
    catch (...) {                                             \
        g_last_error = "unknown exception";                   \
        return ZvecStatus{ZVEC_ERR_INTERNAL, g_last_error.c_str()}; \
    }

#define CATCH_RETURN_NULL                                     \
    }                                                         \
    catch (const std::exception& e) {                         \
        g_last_error = e.what();                              \
        return nullptr;                                       \
    }                                                         \
    catch (...) {                                             \
        g_last_error = "unknown exception";                   \
        return nullptr;                                       \
    }

// ============================================================================
// Error Handling
// ============================================================================

extern "C" {

const char* zvec_last_error_message() {
    return g_last_error.c_str();
}

void zvec_str_free(ZvecStr str) {
    if (str.data) {
        free(str.data);
    }
}

// ============================================================================
// Index Parameters
// ============================================================================

ZvecIndexParams* zvec_index_params_hnsw_create(
    uint32_t metric_type,
    int32_t m,
    int32_t ef_construction,
    uint32_t quantize_type
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::HnswIndexParams>(
        static_cast<zvec::MetricType>(metric_type),
        m,
        ef_construction,
        static_cast<zvec::QuantizeType>(quantize_type)
    );
    auto* wrapper = new ZvecIndexParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecIndexParams* zvec_index_params_ivf_create(
    uint32_t metric_type,
    int32_t n_list,
    int32_t n_iters,
    bool use_soar,
    uint32_t quantize_type
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::IVFIndexParams>(
        static_cast<zvec::MetricType>(metric_type),
        n_list,
        n_iters,
        use_soar,
        static_cast<zvec::QuantizeType>(quantize_type)
    );
    auto* wrapper = new ZvecIndexParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecIndexParams* zvec_index_params_flat_create(
    uint32_t metric_type,
    uint32_t quantize_type
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::FlatIndexParams>(
        static_cast<zvec::MetricType>(metric_type),
        static_cast<zvec::QuantizeType>(quantize_type)
    );
    auto* wrapper = new ZvecIndexParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecIndexParams* zvec_index_params_invert_create(
    bool range_optimized,
    bool extended_wildcard
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::InvertIndexParams>(
        range_optimized,
        extended_wildcard
    );
    auto* wrapper = new ZvecIndexParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

void zvec_index_params_free(ZvecIndexParams* params) {
    delete params;
}

// ============================================================================
// Query Parameters
// ============================================================================

ZvecQueryParams* zvec_query_params_hnsw_create(
    int32_t ef,
    float radius,
    bool is_linear,
    bool is_using_refiner
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::HnswQueryParams>(
        ef, radius, is_linear, is_using_refiner
    );
    auto* wrapper = new ZvecQueryParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecQueryParams* zvec_query_params_ivf_create(
    int32_t n_probes,
    bool is_using_refiner,
    float scale_factor
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::IVFQueryParams>(
        n_probes, is_using_refiner, scale_factor
    );
    auto* wrapper = new ZvecQueryParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecQueryParams* zvec_query_params_flat_create(
    bool is_using_refiner,
    float scale_factor
) {
    CATCH_BEGIN
    auto params = std::make_shared<zvec::FlatQueryParams>(
        is_using_refiner, scale_factor
    );
    auto* wrapper = new ZvecQueryParams();
    wrapper->ptr = params;
    return wrapper;
    CATCH_RETURN_NULL
}

void zvec_query_params_free(ZvecQueryParams* params) {
    delete params;
}

// ============================================================================
// Field Schema
// ============================================================================

ZvecFieldSchema* zvec_field_schema_create(
    ZvecStrView name,
    uint32_t data_type,
    bool nullable,
    uint32_t dimension
) {
    CATCH_BEGIN
    auto schema = std::make_shared<zvec::FieldSchema>(
        to_string(name),
        static_cast<zvec::DataType>(data_type),
        dimension,
        nullable,
        nullptr
    );
    auto* wrapper = new ZvecFieldSchema();
    wrapper->ptr = schema;
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecStatus zvec_field_schema_set_index_params(
    ZvecFieldSchema* schema,
    const ZvecIndexParams* params
) {
    CATCH_BEGIN
    if (!schema || !params) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    schema->ptr->set_index_params(params->ptr);
    return make_status(ZVEC_OK);
    CATCH_END
}

void zvec_field_schema_free(ZvecFieldSchema* schema) {
    delete schema;
}

// ============================================================================
// Collection Schema
// ============================================================================

ZvecCollectionSchema* zvec_collection_schema_create(ZvecStrView name) {
    CATCH_BEGIN
    auto* wrapper = new ZvecCollectionSchema();
    wrapper->schema.set_name(to_string(name));
    return wrapper;
    CATCH_RETURN_NULL
}

ZvecStatus zvec_collection_schema_add_field(
    ZvecCollectionSchema* schema,
    const ZvecFieldSchema* field
) {
    CATCH_BEGIN
    if (!schema || !field) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = schema->schema.add_field(field->ptr);
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_schema_set_max_doc_count_per_segment(
    ZvecCollectionSchema* schema,
    uint64_t count
) {
    CATCH_BEGIN
    if (!schema) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    schema->schema.set_max_doc_count_per_segment(count);
    return make_status(ZVEC_OK);
    CATCH_END
}

void zvec_collection_schema_free(ZvecCollectionSchema* schema) {
    delete schema;
}

// ============================================================================
// Collection
// ============================================================================

ZvecStatus zvec_collection_create_and_open(
    ZvecStrView path,
    const ZvecCollectionSchema* schema,
    const ZvecCollectionOptions* options,
    ZvecCollection** out
) {
    CATCH_BEGIN
    if (!schema || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    zvec::CollectionOptions opts;
    if (options) {
        opts.read_only_ = options->read_only;
        opts.enable_mmap_ = options->enable_mmap;
        opts.max_buffer_size_ = options->max_buffer_size;
    }

    auto result = zvec::Collection::CreateAndOpen(
        to_string(path),
        schema->schema,
        opts
    );

    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecCollection();
    wrapper->ptr = result.value();
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_open(
    ZvecStrView path,
    const ZvecCollectionOptions* options,
    ZvecCollection** out
) {
    CATCH_BEGIN
    if (!out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    zvec::CollectionOptions opts;
    if (options) {
        opts.read_only_ = options->read_only;
        opts.enable_mmap_ = options->enable_mmap;
        opts.max_buffer_size_ = options->max_buffer_size;
    }

    auto result = zvec::Collection::Open(to_string(path), opts);

    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecCollection();
    wrapper->ptr = result.value();
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

void zvec_collection_free(ZvecCollection* coll) {
    delete coll;
}

ZvecStatus zvec_collection_destroy(ZvecCollection* coll) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = coll->ptr->Destroy();
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_path(const ZvecCollection* coll, ZvecStr* out) {
    CATCH_BEGIN
    if (!coll || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto result = coll->ptr->Path();
    if (!result.has_value()) {
        return from_status(result.error());
    }
    *out = to_zvec_str(result.value());
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_stats_string(const ZvecCollection* coll, ZvecStr* out) {
    CATCH_BEGIN
    if (!coll || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto result = coll->ptr->Stats();
    if (!result.has_value()) {
        return from_status(result.error());
    }
    *out = to_zvec_str(result.value().to_string());
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_schema_string(const ZvecCollection* coll, ZvecStr* out) {
    CATCH_BEGIN
    if (!coll || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto result = coll->ptr->Schema();
    if (!result.has_value()) {
        return from_status(result.error());
    }
    *out = to_zvec_str(result.value().to_string());
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_options(const ZvecCollection* coll, ZvecCollectionOptions* out) {
    CATCH_BEGIN
    if (!coll || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto result = coll->ptr->Options();
    if (!result.has_value()) {
        return from_status(result.error());
    }
    out->read_only = result.value().read_only_;
    out->enable_mmap = result.value().enable_mmap_;
    out->max_buffer_size = result.value().max_buffer_size_;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_flush(ZvecCollection* coll) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = coll->ptr->Flush();
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_optimize(ZvecCollection* coll, int32_t concurrency) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    zvec::OptimizeOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->Optimize(opts);
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_create_index(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecIndexParams* params,
    int32_t concurrency
) {
    CATCH_BEGIN
    if (!coll || !params) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    zvec::CreateIndexOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->CreateIndex(to_string(column), params->ptr, opts);
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_drop_index(ZvecCollection* coll, ZvecStrView column) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = coll->ptr->DropIndex(to_string(column));
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_add_column(
    ZvecCollection* coll,
    ZvecStrView column,
    const ZvecFieldSchema* field,
    ZvecStrView expression,
    int32_t concurrency
) {
    CATCH_BEGIN
    if (!coll || !field) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    zvec::AddColumnOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->AddColumn(
        to_string(column),
        field->ptr,
        to_string(expression),
        opts
    );
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_drop_column(ZvecCollection* coll, ZvecStrView column) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = coll->ptr->DropColumn(to_string(column));
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_alter_column(
    ZvecCollection* coll,
    ZvecStrView column,
    ZvecStrView new_name,
    const ZvecFieldSchema* new_schema,
    int32_t concurrency
) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    zvec::AlterColumnOptions opts;
    opts.concurrency_ = concurrency;
    auto status = coll->ptr->AlterColumn(
        to_string(column),
        to_string(new_name),
        new_schema ? new_schema->ptr : nullptr,
        opts
    );
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_insert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    CATCH_BEGIN
    if (!coll || !docs || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::vector<zvec::Doc> doc_vec;
    doc_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (docs[i]) {
            doc_vec.push_back(docs[i]->doc);
        }
    }

    auto result = coll->ptr->Insert(doc_vec);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecStatusList();
    wrapper->statuses = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_upsert(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    CATCH_BEGIN
    if (!coll || !docs || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::vector<zvec::Doc> doc_vec;
    doc_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (docs[i]) {
            doc_vec.push_back(docs[i]->doc);
        }
    }

    auto result = coll->ptr->Upsert(doc_vec);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecStatusList();
    wrapper->statuses = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_update(
    ZvecCollection* coll,
    const ZvecDoc* const* docs,
    size_t count,
    ZvecStatusList** out
) {
    CATCH_BEGIN
    if (!coll || !docs || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::vector<zvec::Doc> doc_vec;
    doc_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (docs[i]) {
            doc_vec.push_back(docs[i]->doc);
        }
    }

    auto result = coll->ptr->Update(doc_vec);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecStatusList();
    wrapper->statuses = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_delete(
    ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecStatusList** out
) {
    CATCH_BEGIN
    if (!coll || !pks || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::vector<std::string> pk_vec;
    pk_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pk_vec.push_back(to_string(pks[i]));
    }

    auto result = coll->ptr->Delete(pk_vec);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecStatusList();
    wrapper->statuses = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_delete_by_filter(ZvecCollection* coll, ZvecStrView filter) {
    CATCH_BEGIN
    if (!coll) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto status = coll->ptr->DeleteByFilter(to_string(filter));
    return from_status(status);
    CATCH_END
}

ZvecStatus zvec_collection_query(
    const ZvecCollection* coll,
    const ZvecVectorQuery* query,
    ZvecDocList** out
) {
    CATCH_BEGIN
    if (!coll || !query || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    auto result = coll->ptr->Query(query->query);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecDocList();
    wrapper->docs = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_group_by_query(
    const ZvecCollection* coll,
    const ZvecGroupByVectorQuery* query,
    ZvecGroupResults** out
) {
    CATCH_BEGIN
    if (!coll || !query || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    auto result = coll->ptr->GroupByQuery(query->query);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecGroupResults();
    wrapper->results = std::move(result.value());
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_collection_fetch(
    const ZvecCollection* coll,
    const ZvecStrView* pks,
    size_t count,
    ZvecDocMap** out
) {
    CATCH_BEGIN
    if (!coll || !pks || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::vector<std::string> pk_vec;
    pk_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pk_vec.push_back(to_string(pks[i]));
    }

    auto result = coll->ptr->Fetch(pk_vec);
    if (!result.has_value()) {
        *out = nullptr;
        return from_status(result.error());
    }

    auto* wrapper = new ZvecDocMap();
    for (auto& [key, doc] : result.value()) {
        wrapper->entries.emplace_back(key, doc);
    }
    *out = wrapper;
    return make_status(ZVEC_OK);
    CATCH_END
}

// ============================================================================
// Document
// ============================================================================

ZvecDoc* zvec_doc_create() {
    CATCH_BEGIN
    return new ZvecDoc();
    CATCH_RETURN_NULL
}

void zvec_doc_free(ZvecDoc* doc) {
    delete doc;
}

void zvec_doc_set_pk(ZvecDoc* doc, ZvecStrView pk) {
    if (doc) {
        doc->doc.set_pk(to_string(pk));
    }
}

void zvec_doc_set_operator(ZvecDoc* doc, uint32_t op) {
    if (doc) {
        doc->doc.set_operator(static_cast<zvec::Operator>(op));
    }
}

ZvecStatus zvec_doc_set_bool(ZvecDoc* doc, ZvecStrView field, bool value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_i32(ZvecDoc* doc, ZvecStrView field, int32_t value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_i64(ZvecDoc* doc, ZvecStrView field, int64_t value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_u32(ZvecDoc* doc, ZvecStrView field, uint32_t value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_u64(ZvecDoc* doc, ZvecStrView field, uint64_t value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_f32(ZvecDoc* doc, ZvecStrView field, float value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_f64(ZvecDoc* doc, ZvecStrView field, double value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), value);
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_string(ZvecDoc* doc, ZvecStrView field, ZvecStrView value) {
    CATCH_BEGIN
    if (!doc) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    doc->doc.set(to_string(field), to_string(value));
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_dense_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const float* data,
    size_t len
) {
    CATCH_BEGIN
    if (!doc || !data) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    std::vector<float> vec(data, data + len);
    doc->doc.set(to_string(field), std::move(vec));
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_dense_f64(
    ZvecDoc* doc,
    ZvecStrView field,
    const double* data,
    size_t len
) {
    CATCH_BEGIN
    if (!doc || !data) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    std::vector<double> vec(data, data + len);
    doc->doc.set(to_string(field), std::move(vec));
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_set_sparse_f32(
    ZvecDoc* doc,
    ZvecStrView field,
    const uint32_t* indices,
    const float* values,
    size_t len
) {
    CATCH_BEGIN
    if (!doc || !indices || !values) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    std::vector<uint32_t> idx_vec(indices, indices + len);
    std::vector<float> val_vec(values, values + len);
    doc->doc.set(to_string(field), std::make_pair(std::move(idx_vec), std::move(val_vec)));
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStr zvec_doc_get_pk(const ZvecDoc* doc) {
    if (!doc) {
        return ZvecStr{nullptr, 0};
    }
    return to_zvec_str(doc->doc.pk());
}

float zvec_doc_get_score(const ZvecDoc* doc) {
    if (!doc) {
        return 0.0f;
    }
    return doc->doc.score();
}

uint64_t zvec_doc_get_doc_id(const ZvecDoc* doc) {
    if (!doc) {
        return 0;
    }
    return doc->doc.doc_id();
}

bool zvec_doc_has_field(const ZvecDoc* doc, ZvecStrView field) {
    if (!doc) {
        return false;
    }
    return doc->doc.has(to_string(field));
}

ZvecStatus zvec_doc_get_string_field(const ZvecDoc* doc, ZvecStrView field, ZvecStr* out) {
    CATCH_BEGIN
    if (!doc || !out) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }
    auto result = doc->doc.get<std::string>(to_string(field));
    if (!result.has_value()) {
        out->data = nullptr;
        out->len = 0;
        return make_status(ZVEC_ERR_NOT_FOUND, "field not found or wrong type");
    }
    *out = to_zvec_str(result.value());
    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStatus zvec_doc_get_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* out_data_type,
    ZvecArrayView* out_data
) {
    CATCH_BEGIN
    if (!doc || !out_data_type || !out_data) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::string field_name = to_string(field);

    // Try float vector first (copy into thread-local storage to keep data alive).
    auto f32_result = doc->doc.get<std::vector<float>>(field_name);
    if (f32_result.has_value()) {
        thread_local std::vector<float> temp_f32;
        temp_f32 = f32_result.value();
        *out_data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP32);
        out_data->data = temp_f32.data();
        out_data->len = temp_f32.size();
        return make_status(ZVEC_OK);
    }

    // Try double vector
    auto f64_result = doc->doc.get<std::vector<double>>(field_name);
    if (f64_result.has_value()) {
        thread_local std::vector<double> temp_f64;
        temp_f64 = f64_result.value();
        *out_data_type = static_cast<uint32_t>(zvec::DataType::VECTOR_FP64);
        out_data->data = temp_f64.data();
        out_data->len = temp_f64.size();
        return make_status(ZVEC_OK);
    }

    return make_status(ZVEC_ERR_NOT_FOUND, "vector field not found");
    CATCH_END
}

ZvecStatus zvec_doc_get_sparse_vector_field(
    const ZvecDoc* doc,
    ZvecStrView field,
    uint32_t* out_data_type,
    ZvecArrayView* out_indices,
    ZvecArrayView* out_values
) {
    CATCH_BEGIN
    if (!doc || !out_data_type || !out_indices || !out_values) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::string field_name = to_string(field);

    using SparseF32 = std::pair<std::vector<uint32_t>, std::vector<float>>;
    auto result = doc->doc.get<SparseF32>(field_name);
    if (result.has_value()) {
        thread_local std::vector<uint32_t> temp_indices;
        thread_local std::vector<float> temp_values;
        const auto& [indices, values] = result.value();
        temp_indices = indices;
        temp_values = values;
        *out_data_type = static_cast<uint32_t>(zvec::DataType::SPARSE_VECTOR_FP32);
        out_indices->data = temp_indices.data();
        out_indices->len = temp_indices.size();
        out_values->data = temp_values.data();
        out_values->len = temp_values.size() * sizeof(float);
        return make_status(ZVEC_OK);
    }

    return make_status(ZVEC_ERR_NOT_FOUND, "sparse vector field not found");
    CATCH_END
}

ZvecStatus zvec_doc_get_field_type(
    const ZvecDoc* doc,
    ZvecStrView field,
    bool* out_found,
    bool* out_is_null,
    uint32_t* out_data_type
) {
    CATCH_BEGIN
    if (!doc || !out_found || !out_is_null || !out_data_type) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "null pointer");
    }

    std::string field_name = to_string(field);
    *out_found = doc->doc.has(field_name);
    *out_is_null = doc->doc.is_null(field_name);
    *out_data_type = static_cast<uint32_t>(zvec::DataType::UNDEFINED);
    if (!*out_found || *out_is_null) {
        return make_status(ZVEC_OK);
    }

    auto match_type = [&](auto tag, zvec::DataType dtype) -> bool {
        using T = decltype(tag);
        auto result = doc->doc.get_field<T>(field_name);
        if (result.status() == zvec::Doc::FieldGetStatus::SUCCESS) {
            *out_data_type = static_cast<uint32_t>(dtype);
            return true;
        }
        return false;
    };

    // Scalar types
    if (match_type(bool{}, zvec::DataType::BOOL)) return make_status(ZVEC_OK);
    if (match_type(int32_t{}, zvec::DataType::INT32)) return make_status(ZVEC_OK);
    if (match_type(uint32_t{}, zvec::DataType::UINT32)) return make_status(ZVEC_OK);
    if (match_type(int64_t{}, zvec::DataType::INT64)) return make_status(ZVEC_OK);
    if (match_type(uint64_t{}, zvec::DataType::UINT64)) return make_status(ZVEC_OK);
    if (match_type(float{}, zvec::DataType::FLOAT)) return make_status(ZVEC_OK);
    if (match_type(double{}, zvec::DataType::DOUBLE)) return make_status(ZVEC_OK);
    if (match_type(std::string{}, zvec::DataType::STRING)) return make_status(ZVEC_OK);

    // Dense vectors
    if (match_type(std::vector<float>{}, zvec::DataType::VECTOR_FP32)) return make_status(ZVEC_OK);
    if (match_type(std::vector<double>{}, zvec::DataType::VECTOR_FP64)) return make_status(ZVEC_OK);
    if (match_type(std::vector<zvec::float16_t>{}, zvec::DataType::VECTOR_FP16)) return make_status(ZVEC_OK);
    if (match_type(std::vector<int8_t>{}, zvec::DataType::VECTOR_INT8)) return make_status(ZVEC_OK);
    if (match_type(std::vector<int16_t>{}, zvec::DataType::VECTOR_INT16)) return make_status(ZVEC_OK);

    // Array types
    if (match_type(std::vector<bool>{}, zvec::DataType::ARRAY_BOOL)) return make_status(ZVEC_OK);
    if (match_type(std::vector<int32_t>{}, zvec::DataType::ARRAY_INT32)) return make_status(ZVEC_OK);
    if (match_type(std::vector<int64_t>{}, zvec::DataType::ARRAY_INT64)) return make_status(ZVEC_OK);
    if (match_type(std::vector<uint32_t>{}, zvec::DataType::ARRAY_UINT32)) return make_status(ZVEC_OK);
    if (match_type(std::vector<uint64_t>{}, zvec::DataType::ARRAY_UINT64)) return make_status(ZVEC_OK);
    if (match_type(std::vector<std::string>{}, zvec::DataType::ARRAY_STRING)) return make_status(ZVEC_OK);

    // Sparse vectors
    using SparseF32 = std::pair<std::vector<uint32_t>, std::vector<float>>;
    using SparseF16 = std::pair<std::vector<uint32_t>, std::vector<zvec::float16_t>>;
    if (match_type(SparseF32{}, zvec::DataType::SPARSE_VECTOR_FP32)) return make_status(ZVEC_OK);
    if (match_type(SparseF16{}, zvec::DataType::SPARSE_VECTOR_FP16)) return make_status(ZVEC_OK);

    return make_status(ZVEC_OK);
    CATCH_END
}

ZvecStr zvec_doc_to_string(const ZvecDoc* doc) {
    if (!doc) {
        return ZvecStr{nullptr, 0};
    }
    return to_zvec_str(doc->doc.to_string());
}

ZvecStr zvec_doc_to_detail_string(const ZvecDoc* doc) {
    if (!doc) {
        return ZvecStr{nullptr, 0};
    }
    return to_zvec_str(doc->doc.to_detail_string());
}

// ============================================================================
// Document List
// ============================================================================

size_t zvec_doc_list_len(const ZvecDocList* list) {
    if (!list) {
        return 0;
    }
    return list->docs.size();
}

const ZvecDoc* zvec_doc_list_get(const ZvecDocList* list, size_t index) {
    if (!list || index >= list->docs.size()) {
        return nullptr;
    }
    // Return a pointer to a temporary wrapper - the lifetime is tied to the list
    // This is a simplification; in production we'd need better memory management
    thread_local ZvecDoc temp_doc;
    if (list->docs[index]) {
        temp_doc.doc = *list->docs[index];
    }
    return &temp_doc;
}

void zvec_doc_list_free(ZvecDocList* list) {
    delete list;
}

// ============================================================================
// Document Map
// ============================================================================

size_t zvec_doc_map_len(const ZvecDocMap* map) {
    if (!map) {
        return 0;
    }
    return map->entries.size();
}

ZvecStr zvec_doc_map_get_key(const ZvecDocMap* map, size_t index) {
    if (!map || index >= map->entries.size()) {
        return ZvecStr{nullptr, 0};
    }
    return to_zvec_str(map->entries[index].first);
}

const ZvecDoc* zvec_doc_map_get_value(const ZvecDocMap* map, size_t index) {
    if (!map || index >= map->entries.size()) {
        return nullptr;
    }
    thread_local ZvecDoc temp_doc;
    if (map->entries[index].second) {
        temp_doc.doc = *map->entries[index].second;
    }
    return &temp_doc;
}

void zvec_doc_map_free(ZvecDocMap* map) {
    delete map;
}

// ============================================================================
// Status List
// ============================================================================

size_t zvec_status_list_len(const ZvecStatusList* list) {
    if (!list) {
        return 0;
    }
    return list->statuses.size();
}

ZvecStatus zvec_status_list_get(const ZvecStatusList* list, size_t index) {
    if (!list || index >= list->statuses.size()) {
        return make_status(ZVEC_ERR_INVALID_ARGUMENT, "invalid index");
    }
    return from_status(list->statuses[index]);
}

void zvec_status_list_free(ZvecStatusList* list) {
    delete list;
}

// ============================================================================
// Vector Query
// ============================================================================

ZvecVectorQuery* zvec_vector_query_create(
    ZvecStrView field_name,
    const void* dense_data,
    size_t dense_len,
    const uint32_t* sparse_indices,
    const void* sparse_values,
    size_t sparse_len,
    int32_t topk
) {
    CATCH_BEGIN
    auto* wrapper = new ZvecVectorQuery();
    wrapper->query.field_name_ = to_string(field_name);
    wrapper->query.topk_ = topk;

    if (dense_data && dense_len > 0) {
        wrapper->query.query_vector_.assign(
            static_cast<const char*>(dense_data),
            dense_len
        );
    }

    if (sparse_indices && sparse_values && sparse_len > 0) {
        wrapper->query.query_sparse_indices_.assign(
            reinterpret_cast<const char*>(sparse_indices),
            sparse_len * sizeof(uint32_t)
        );
        wrapper->query.query_sparse_values_.assign(
            static_cast<const char*>(sparse_values),
            sparse_len * sizeof(float)
        );
    }

    return wrapper;
    CATCH_RETURN_NULL
}

void zvec_vector_query_set_filter(ZvecVectorQuery* query, ZvecStrView filter) {
    if (query) {
        query->query.filter_ = to_string(filter);
    }
}

void zvec_vector_query_set_output_fields(
    ZvecVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
) {
    if (!query) return;

    if (!fields || count == 0) {
        query->query.output_fields_ = std::vector<std::string>();
    } else {
        std::vector<std::string> field_vec;
        field_vec.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            field_vec.push_back(to_string(fields[i]));
        }
        query->query.output_fields_ = std::move(field_vec);
    }
}

void zvec_vector_query_set_query_params(ZvecVectorQuery* query, const ZvecQueryParams* params) {
    if (query && params) {
        query->query.query_params_ = params->ptr;
    }
}

void zvec_vector_query_set_include_vector(ZvecVectorQuery* query, bool include) {
    if (query) {
        query->query.include_vector_ = include;
    }
}

void zvec_vector_query_set_include_doc_id(ZvecVectorQuery* query, bool include) {
    if (query) {
        query->query.include_doc_id_ = include;
    }
}

void zvec_vector_query_free(ZvecVectorQuery* query) {
    delete query;
}

// ============================================================================
// Group By Vector Query
// ============================================================================

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
) {
    CATCH_BEGIN
    auto* wrapper = new ZvecGroupByVectorQuery();
    wrapper->query.field_name_ = to_string(field_name);
    wrapper->query.group_by_field_name_ = to_string(group_by_field);
    wrapper->query.group_count_ = group_count;
    wrapper->query.group_topk_ = group_topk;

    if (dense_data && dense_len > 0) {
        wrapper->query.query_vector_.assign(
            static_cast<const char*>(dense_data),
            dense_len
        );
    }

    if (sparse_indices && sparse_values && sparse_len > 0) {
        wrapper->query.query_sparse_indices_.assign(
            reinterpret_cast<const char*>(sparse_indices),
            sparse_len * sizeof(uint32_t)
        );
        wrapper->query.query_sparse_values_.assign(
            static_cast<const char*>(sparse_values),
            sparse_len * sizeof(float)
        );
    }

    return wrapper;
    CATCH_RETURN_NULL
}

void zvec_group_by_vector_query_set_filter(ZvecGroupByVectorQuery* query, ZvecStrView filter) {
    if (query) {
        query->query.filter_ = to_string(filter);
    }
}

void zvec_group_by_vector_query_set_output_fields(
    ZvecGroupByVectorQuery* query,
    const ZvecStrView* fields,
    size_t count
) {
    if (!query) return;

    if (!fields || count == 0) {
        query->query.output_fields_ = std::vector<std::string>();
    } else {
        std::vector<std::string> field_vec;
        field_vec.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            field_vec.push_back(to_string(fields[i]));
        }
        query->query.output_fields_ = std::move(field_vec);
    }
}

void zvec_group_by_vector_query_set_query_params(
    ZvecGroupByVectorQuery* query,
    const ZvecQueryParams* params
) {
    if (query && params) {
        query->query.query_params_ = params->ptr;
    }
}

void zvec_group_by_vector_query_set_include_vector(ZvecGroupByVectorQuery* query, bool include) {
    if (query) {
        query->query.include_vector_ = include;
    }
}

void zvec_group_by_vector_query_free(ZvecGroupByVectorQuery* query) {
    delete query;
}

// ============================================================================
// Group Results
// ============================================================================

size_t zvec_group_results_len(const ZvecGroupResults* results) {
    if (!results) {
        return 0;
    }
    return results->results.size();
}

const ZvecGroupResult* zvec_group_results_get(const ZvecGroupResults* results, size_t index) {
    if (!results || index >= results->results.size()) {
        return nullptr;
    }
    // Return a stable pointer
    thread_local ZvecGroupResult temp_result;
    temp_result.result = &results->results[index];
    temp_result.doc_list.docs.clear();
    for (const auto& doc : results->results[index].docs_) {
        temp_result.doc_list.docs.push_back(std::make_shared<zvec::Doc>(doc));
    }
    return &temp_result;
}

void zvec_group_results_free(ZvecGroupResults* results) {
    delete results;
}

ZvecStr zvec_group_result_get_group_value(const ZvecGroupResult* result) {
    if (!result || !result->result) {
        return ZvecStr{nullptr, 0};
    }
    return to_zvec_str(result->result->group_by_value_);
}

const ZvecDocList* zvec_group_result_get_docs(const ZvecGroupResult* result) {
    if (!result) {
        return nullptr;
    }
    return &result->doc_list;
}

}  // extern "C"

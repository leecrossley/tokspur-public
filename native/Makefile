# No package manager, reference checkout or prebuilt engine library is used.
CXX ?= c++
CPPFLAGS = -Iinclude
CXXFLAGS ?= -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Werror
M5_HEADLINE_SCREEN ?= 0
VERIFY_NARROW_ROWS ?= 0
GDN_COOPERATIVE4 ?= 0
ifeq ($(GDN_COOPERATIVE4),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_GDN_COOPERATIVE4=1
override METALFLAGS += -DTOKSPUR_PRIVATE_GDN_COOPERATIVE4=1
else ifneq ($(GDN_COOPERATIVE4),0)
$(error GDN_COOPERATIVE4 must be 0 or 1)
endif
VERIFY_GDN_ROW_BATCH ?= 1
ifeq ($(VERIFY_GDN_ROW_BATCH),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_VERIFY_GDN_ROW_BATCH=1
else ifneq ($(VERIFY_GDN_ROW_BATCH),0)
$(error VERIFY_GDN_ROW_BATCH must be 0 or 1)
endif
ifeq ($(M5_HEADLINE_SCREEN),1)
override M5_CPU_CANDIDATES := 1
override CXXFLAGS += -DTOKSPUR_PRIVATE_M5_RESOURCE_SCREEN=1
override METALFLAGS += -DTOKSPUR_PRIVATE_PLANAR_COLUMN32=1 -DTOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE=1 -DTOKSPUR_PRIVATE_PLANAR_GU_STAGE64=1 -DTOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE=1
else ifneq ($(M5_HEADLINE_SCREEN),0)
$(error M5_HEADLINE_SCREEN must be 0 or 1)
endif
ifeq ($(VERIFY_NARROW_ROWS),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_VERIFY_NARROW_ROWS=1
override METALFLAGS += -DTOKSPUR_PRIVATE_VERIFY_NARROW_ROWS=1
else ifneq ($(VERIFY_NARROW_ROWS),0)
$(error VERIFY_NARROW_ROWS must be 0 or 1)
endif
PARALLEL_PLANAR_UNION ?= 0
ifeq ($(PARALLEL_PLANAR_UNION),1)
override METALFLAGS += -DTOKSPUR_PRIVATE_PARALLEL_PLANAR_UNION=1
else ifneq ($(PARALLEL_PLANAR_UNION),0)
$(error PARALLEL_PLANAR_UNION must be 0 or 1)
endif
M5_CPU_CANDIDATES ?= 0
Q8_IMPORT_ISOLATION ?= 0
PACK_SHARED_MAPPING ?= 0
# Opt-in bounded original-source reads and clean-page hints during v2/v3 authentication.
PACK_AUTH_RECLAIM ?= 0
CACHE_PRESSURE_RECLAIM ?= 0
MEMORY_PLAN_REPORT ?= 0
SHARED_PREFIX_EVICTION_COUNTERS ?= 0
# Optional host PLE row-cache limit; 0 preserves 512 MiB (256 MiB in the M5 screen).
PLE_CACHE_MIB ?= 0
# Optional duplicate Q8 Tile64 storage; original-layout consumers remain available.
Q8_TILE64_CACHE ?= 1
PIPELINE_LOOKUP ?= 1
DFLASH2_CACHE_WORK ?= 0
# Batched prefill top-10 routing and mixture; 0 dispatches each token row separately.
PREFILL_BATCH_TOP10 ?= 1
PREFILL_BATCH_MIXTURE ?= 1
ifeq ($(PREFILL_BATCH_TOP10):$(PREFILL_BATCH_MIXTURE),1:1)
VERIFY_NAX_BATCH_TOP10_MIXTURE ?= 1
else
VERIFY_NAX_BATCH_TOP10_MIXTURE ?= 0
endif
NATIVE_CONFIG ?= carrier
ifeq ($(Q8_TILE64_CACHE),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_Q8_TILE64_CACHE=1
else ifeq ($(Q8_TILE64_CACHE),0)
override CXXFLAGS += -DTOKSPUR_PRIVATE_Q8_TILE64_CACHE=0
else
$(error Q8_TILE64_CACHE must be 0 or 1)
endif
ifeq ($(PREFILL_BATCH_TOP10),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PREFILL_BATCH_TOP10=1
override METALFLAGS += -DTOKSPUR_PRIVATE_PREFILL_BATCH_TOP10=1
else ifneq ($(PREFILL_BATCH_TOP10),0)
$(error PREFILL_BATCH_TOP10 must be 0 or 1)
endif
ifeq ($(PREFILL_BATCH_MIXTURE),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE=1
override METALFLAGS += -DTOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE=1
else ifneq ($(PREFILL_BATCH_MIXTURE),0)
$(error PREFILL_BATCH_MIXTURE must be 0 or 1)
endif
ifeq ($(VERIFY_NAX_BATCH_TOP10_MIXTURE),1)
ifneq ($(PREFILL_BATCH_TOP10):$(PREFILL_BATCH_MIXTURE),1:1)
$(error VERIFY_NAX_BATCH_TOP10_MIXTURE requires PREFILL_BATCH_TOP10=1 and PREFILL_BATCH_MIXTURE=1)
endif
override CXXFLAGS += -DTOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE=1
else ifneq ($(VERIFY_NAX_BATCH_TOP10_MIXTURE),0)
$(error VERIFY_NAX_BATCH_TOP10_MIXTURE must be 0 or 1)
endif
ifeq ($(Q8_IMPORT_ISOLATION),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_Q8_IMPORT_ISOLATION=1
else ifneq ($(Q8_IMPORT_ISOLATION),0)
$(error Q8_IMPORT_ISOLATION must be 0 or 1)
endif
ifeq ($(PACK_SHARED_MAPPING),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PACK_SHARED_MAPPING=1
else ifneq ($(PACK_SHARED_MAPPING),0)
$(error PACK_SHARED_MAPPING must be 0 or 1)
endif
ifeq ($(PACK_AUTH_RECLAIM),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PACK_AUTH_RECLAIM=1
else ifneq ($(PACK_AUTH_RECLAIM),0)
$(error PACK_AUTH_RECLAIM must be 0 or 1)
endif
ifeq ($(CACHE_PRESSURE_RECLAIM),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM=1
else ifneq ($(CACHE_PRESSURE_RECLAIM),0)
$(error CACHE_PRESSURE_RECLAIM must be 0 or 1)
endif
ifeq ($(SHARED_PREFIX_EVICTION_COUNTERS),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS=1
else ifneq ($(SHARED_PREFIX_EVICTION_COUNTERS),0)
$(error SHARED_PREFIX_EVICTION_COUNTERS must be 0 or 1)
endif
ifeq ($(MEMORY_PLAN_REPORT),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_MEMORY_PLAN_REPORT=1
else ifneq ($(MEMORY_PLAN_REPORT),0)
$(error MEMORY_PLAN_REPORT must be 0 or 1)
endif
ifeq ($(PLE_CACHE_MIB),0)
else ifeq ($(PLE_CACHE_MIB),256)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PLE_CACHE_MIB=256
else ifeq ($(PLE_CACHE_MIB),512)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PLE_CACHE_MIB=512
else
$(error PLE_CACHE_MIB must be 0 (automatic), 256 or 512)
endif
ifeq ($(PIPELINE_LOOKUP),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_PIPELINE_LOOKUP=1
else ifneq ($(PIPELINE_LOOKUP),0)
$(error PIPELINE_LOOKUP must be 0 or 1)
endif
ifeq ($(DFLASH2_CACHE_WORK),1)
override CXXFLAGS += -DTOKSPUR_PRIVATE_DFLASH2_CACHE_WORK=1
else ifneq ($(DFLASH2_CACHE_WORK),0)
$(error DFLASH2_CACHE_WORK must be 0 or 1)
endif
ifneq ($(M5_CPU_CANDIDATES),0)
ifneq ($(M5_CPU_CANDIDATES),1)
$(error Unsupported M5_CPU_CANDIDATES '$(M5_CPU_CANDIDATES)'; expected 0 or 1)
endif
endif
ifeq ($(SANITIZE),1)
override CXXFLAGS += -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
endif
BUILD ?= $(if $(filter carrier-diagnostic,$(NATIVE_CONFIG)),../build/carrier-diagnostic,../build/owned-native)
SANITIZE_BUILD ?= ../build/owned-sanitize
CI_SANITIZE_BUILD ?= ../build/ci-sanitize
HTTP_FAST_MATH_BUILD ?= ../build/http-fast-math
OBJ_ROOT = $(BUILD)/obj
OBJDIR = $(OBJ_ROOT)/$(NATIVE_CONFIG)
COMMON = measured_controller gguf numeric tokenizer unicode json chat operators model routing gdn_math recurrent_reference attention_math attention_reference memory ple_math sampling response response_protocol q8_layout q8_many gdn_tile serve_protocol http owned_worker prompt_lookup transcript_replay mtp_wire serve_verify8 expert_lanes lane_sidecar derived_pack planar_pack_v3 pack_v3_cli ssd_checkpoint
OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(COMMON)))
# The CPU-only host telemetry collector links only into its own test and diagnostic.
TELEMETRY_OBJECTS = $(OBJDIR)/host_telemetry.o
CPU_TESTS = pipeline_key dflash2_checkpoint shared_prefix_restore speculative_controls gguf numeric tokenizer unicode json chat operators model routing gdn reference attention_math memory sampling response response_protocol q8_layout q8_many gdn_tile recurrent_timing_protocol recurrent_replay_timing_protocol serve_protocol http owned_worker prompt_lookup transcript_replay derived_pack host_telemetry
CPU_TESTS += carrier_diagnostic measured_controller nax_split_policy ssd_checkpoint literal_markers shared_prefix_eviction
PATTERN_TESTS = $(filter-out host_telemetry speculative_controls ssd_checkpoint,$(CPU_TESTS))
TEST_BINS = $(addprefix $(BUILD)/test-,$(CPU_TESTS))
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
TELEMETRY_OBJECTS += $(OBJDIR)/host_telemetry_apple.o
HOST_TELEMETRY_BINARY = $(BUILD)/tokspur-host-telemetry
endif

CARRIER_DEFS = \
	-DTOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8=1 \
	-DTOKSPUR_PRIVATE_QSA_STAGE_V8=1 \
	-DTOKSPUR_PRIVATE_HTTP_BRANCH8=1 \
	-DTOKSPUR_PRIVATE_VERIFY8_BRANCH=1 \
	-DTOKSPUR_PRIVATE_VERIFY_EIGHT=1 \
	-DTOKSPUR_PRIVATE_HTTP_VERIFY8=1 \
	-DTOKSPUR_PRIVATE_HC_EXACT=1 \
	-DTOKSPUR_PRIVATE_WIDE_Q8_EXACT=1 \
	-DTOKSPUR_PRIVATE_WIDE_Q8_MANY=1 \
	-DTOKSPUR_PRIVATE_HTTP_FAST_MATH=1 \
	-DTOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL=1 \
	-DTOKSPUR_PRIVATE_PLE_GATE_COMPAT=1 \
	-DTOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL=1 \
	-DTOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL=1 \
	-DTOKSPUR_PRIVATE_VERIFY_QSA_ROWS=1 \
	-DTOKSPUR_PRIVATE_KNOWN_QSA_ROWS=1 \
	-DTOKSPUR_PRIVATE_NATIVE_MTP=1 \
	-DTOKSPUR_PRIVATE_LANE_LAYOUT=1 \
	-DTOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2=1 \
	-DTOKSPUR_PRIVATE_BOUNDED_TOPK=1 \
	-DTOKSPUR_PRIVATE_CONCURRENT_DISPATCH=1 \
	-DTOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT=1 \
	-DTOKSPUR_PRIVATE_MTP_OUTPUT64=1 \
	-DTOKSPUR_PRIVATE_HTTP_PREFILL512=1 \
	-DTOKSPUR_PRIVATE_MTP_PRIMING_BATCH=1 \
	-DTOKSPUR_PRIVATE_PREFILL_CHUNK=1 \
	-DTOKSPUR_PRIVATE_GDN_ELEMENTWISE_BATCH=1 \
	-DTOKSPUR_PRIVATE_Q8_MPP=1 \
	-DTOKSPUR_PRIVATE_GROUPED_MPP=1 \
	-DTOKSPUR_PRIVATE_GROUPED_MM=1 \
	-DTOKSPUR_PRIVATE_GDN_CHUNK=1 \
	-DTOKSPUR_PRIVATE_QSA_PREFILL=1 \
	-DTOKSPUR_PRIVATE_CHUNKED_PREFILL=1 \
	-DTOKSPUR_PRIVATE_GROUPED_MM_F32=1 \
	-DTOKSPUR_PRIVATE_GDN_LEGACY_BATCH=1 \
	-DTOKSPUR_PRIVATE_Q8_THREE_ROWS=1 \
	-DTOKSPUR_PRIVATE_RAW_DECODE=1 \
	-DTOKSPUR_PRIVATE_DENSE_Q8_STAGED=1 \
	-DTOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32=1 \
	-DTOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS=1 \
	-DTOKSPUR_PRIVATE_LOGIT_TRACE=1 \
	-DTOKSPUR_PRIVATE_HC_ROWS=1 \
	-DTOKSPUR_PRIVATE_HTTP_HC_ROWS=1 \
	-DTOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH=1 \
	-DTOKSPUR_PRIVATE_MOE_ROUTER_WINDOW=1 \
	-DTOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW=1 \
	-DTOKSPUR_PRIVATE_HC_DOWN_WINDOW=1 \
	-DTOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW=1 \
	-DTOKSPUR_PRIVATE_QSA_SPARSE=1 \
	-DTOKSPUR_PRIVATE_HTTP_QSA_SPARSE=1 \
	-DTOKSPUR_PRIVATE_GDN_PORTED=1 \
	-DTOKSPUR_PRIVATE_HTTP_GDN_PORTED=1 \
	-DTOKSPUR_PRIVATE_SPARSE_RESIDUAL=1


CARRIER_HTTP_OBJECTS = $(addprefix $(OBJDIR)/,mtp.o mtp_sampling.o sampled_path.o carrier_owned_worker.o carrier_serve_protocol.o)
CARRIER_BASE_OBJECTS = $(filter-out %owned_worker.o %serve_protocol.o,$(OBJECTS))
CARRIER_ALL_OBJECTS = $(CARRIER_BASE_OBJECTS) $(CARRIER_HTTP_OBJECTS)

HTTP_FAST_MATH_DEFS = \
	-DTOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8=1 \
	-DTOKSPUR_PRIVATE_QSA_STAGE_V8=1 \
	-DTOKSPUR_PRIVATE_HTTP_BRANCH8=1 \
	-DTOKSPUR_PRIVATE_VERIFY8_BRANCH=1 \
	-DTOKSPUR_PRIVATE_VERIFY_EIGHT=1 \
	-DTOKSPUR_PRIVATE_HTTP_VERIFY8=1 \
	-DTOKSPUR_PRIVATE_HC_EXACT=1 \
	-DTOKSPUR_PRIVATE_WIDE_Q8_EXACT=1 \
	-DTOKSPUR_PRIVATE_WIDE_Q8_MANY=1 \
	-DTOKSPUR_PRIVATE_HTTP_FAST_MATH=1 \
	-DTOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL=1 \
	-DTOKSPUR_PRIVATE_PLE_GATE_COMPAT=1 \
	-DTOKSPUR_PRIVATE_BATCH_PHASE_CLOCK=1

ifeq ($(NATIVE_CONFIG),carrier)
ACTIVE_DEFS = $(CARRIER_DEFS)
ACTIVE_HTTP_OBJECTS = $(CARRIER_ALL_OBJECTS)
else ifeq ($(NATIVE_CONFIG),carrier-diagnostic)
ACTIVE_DEFS = $(CARRIER_DEFS) -DTOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC=1 -DTOKSPUR_PRIVATE_BATCH_PHASE_CLOCK=1 -DTOKSPUR_PRIVATE_DIAGNOSTIC_BATCH_CLOCK=1
ACTIVE_HTTP_OBJECTS = $(CARRIER_ALL_OBJECTS)
else ifeq ($(NATIVE_CONFIG),http-fast-math)
ACTIVE_DEFS = $(HTTP_FAST_MATH_DEFS)
ACTIVE_HTTP_OBJECTS = $(CARRIER_BASE_OBJECTS) $(CARRIER_HTTP_OBJECTS)
else ifeq ($(NATIVE_CONFIG),native-plain-loads)
ACTIVE_DEFS = -DTOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS=1
ACTIVE_HTTP_OBJECTS = $(OBJECTS)
else ifeq ($(NATIVE_CONFIG),base)
ACTIVE_DEFS =
ACTIVE_HTTP_OBJECTS = $(OBJECTS)
else
$(error Unsupported NATIVE_CONFIG '$(NATIVE_CONFIG)'; expected carrier, carrier-diagnostic, http-fast-math, native-plain-loads or base)
endif

CPPFLAGS += $(ACTIVE_DEFS)
CARRIER_CPPFLAGS = $(CPPFLAGS)
ENABLE_METAL ?= $(if $(filter Darwin,$(UNAME_S)),1,0)
ifeq ($(ENABLE_METAL),1)
CPPFLAGS += -DTOKSPUR_HAS_METAL=1
BASE_METAL_OBJECTS = metal.o graph.o moe.o recurrent.o recurrent_compare.o recurrent_check.o gdn_core_tile.o gdn_core_compare.o gdn_core_tile_check.o recurrent_tile.o recurrent_tile_compare.o recurrent_tile_check.o recurrent_timing.o recurrent_replay_timing.o attention.o attention_compare.o attention_check.o ple.o memory_check.o stack.o target.o mapping_check.o layout_check.o q8_many_check.o ffn_check.o text.o response_cli.o
CARRIER_OBJECTS = mtp.o sampled_path.o mtp_sampling.o mtp_head.o dflash2_drafter.o q8_mpp.o grouped_mpp.o grouped_mm.o gdn_chunk_math.o qsa_prefill_plan.o qsa_prefill_ops.o moe_mm_many_plan.o moe_many.o gdn_chunk.o qsa_prefill.o prefill_residual.o moe_mm_many.o prefill_gdn.o prefill_attention.o grouped_mm_f32.o gdn_legacy_batch_plan.o q8_three_rows.o moe_q8_three_rows.o raw_decode_component.o logit_trace.o attention_tile.o verify_nax.o
ifneq ($(filter carrier carrier-diagnostic,$(NATIVE_CONFIG)),)
METAL_OBJECTS = $(addprefix $(OBJDIR)/,$(BASE_METAL_OBJECTS) $(CARRIER_OBJECTS))
SCALAR_WORKER_BINARY = $(BUILD)/scalar-channel-target
else
METAL_OBJECTS = $(addprefix $(OBJDIR)/,$(BASE_METAL_OBJECTS))
endif
METAL_FLAGS = -framework Foundation -framework Metal -pthread
METAL_LIBRARY = $(BUILD)/tokspur-owned.metallib
SERVER_BINARY = $(BUILD)/tokspur-server
endif

.PHONY: FORCE
ifeq ($(M5_CPU_CANDIDATES),1)
override CPPFLAGS += -DTOKSPUR_PRIVATE_M5_CPU_CANDIDATES=1
endif
TRACKED_CFLAGS := $(NATIVE_CONFIG) : $(ENABLE_METAL) : m5-cpu-candidates=$(M5_CPU_CANDIDATES) : $(CXX) : $(CPPFLAGS) : $(CXXFLAGS)
TRACKED_METALFLAGS := $(NATIVE_CONFIG) : metal3.2 : metal4.0 : $(METALFLAGS)
# Decide before rewriting the stamp. FORCE also handles rapid flag changes on
# Apple's make, whose timestamp comparisons can have one-second resolution.
OBJECT_POLICY_PREREQUISITE := $(shell echo '$(TRACKED_CFLAGS)' | cmp -s - '$(OBJDIR)/.cflags' 2>/dev/null || echo FORCE)
# These targets can be built in separate invocations. Compare each artifact's
# last successful policy even if another target has updated the shared stamp.
metal_policy_prerequisite = $(shell echo '$(TRACKED_METALFLAGS)' | cmp -s - '$(1).metalflags' 2>/dev/null || echo FORCE)

$(BUILD)/.cflags: FORCE | $(BUILD)
	@echo '$(TRACKED_CFLAGS)' | cmp -s - $@ 2>/dev/null || echo '$(TRACKED_CFLAGS)' > $@

# Keep the actual object cache's policy even when separate output directories
# deliberately share OBJDIR. A/B/A builds must not reuse B's newer objects under
# A's unchanged, older output manifest.
$(OBJDIR)/.cflags: FORCE | $(OBJDIR)
	@echo '$(TRACKED_CFLAGS)' | cmp -s - $@ 2>/dev/null || echo '$(TRACKED_CFLAGS)' > $@

$(BUILD)/.metalflags: FORCE | $(BUILD)
	@echo '$(TRACKED_METALFLAGS)' | cmp -s - $@ 2>/dev/null || echo '$(TRACKED_METALFLAGS)' > $@

.PHONY: all tokspur-native tokspur-server tokspur-owned.metallib test test-metal carrier-test-metal sanitize fuzz coverage-fuzz clean ci-components http-fast-math native-plain-loads-component rust-check test-sparse_residual_dual_order test-concurrent_abba_benchmark test-m4_s3_abba_benchmark
all: $(BUILD)/tokspur-native $(SCALAR_WORKER_BINARY) $(METAL_LIBRARY) $(SERVER_BINARY) $(HOST_TELEMETRY_BINARY)
tokspur-native: $(BUILD)/tokspur-native
tokspur-server: $(SERVER_BINARY)
tokspur-owned.metallib: $(METAL_LIBRARY)

# Standalone CPU-only worker-name/SDK binding: no model or Metal dependencies.
.PHONY: worker-identity-cpu
worker-identity-cpu: $(BUILD)/test-worker_identity
$(BUILD)/test-worker_identity: tests/test_worker_identity.cpp include/tokspur/worker_identity.hpp $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" -o "$@"

# CPU/Foundation only: no Metal objects, shaders, model descriptors or server.
.PHONY: transcript-replay-cpu
transcript-replay-cpu: $(BUILD)/tokspur-transcript-replay $(BUILD)/test-transcript_replay
$(BUILD)/tokspur-transcript-replay: tests/transcript_replay_main.cpp $(OBJDIR)/text.o $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJDIR)/text.o $(OBJECTS) -framework Foundation -o "$@"

# Explicit, CPU-only metadata component. Deliberately absent from COMMON, all,
# default tests, the native target and serving. No Metal/Foundation link.
MTP_OBJECTS = $(addprefix $(OBJDIR)/,gguf.o numeric.o unicode.o json.o model.o mtp.o planar_pack_v3.o derived_pack.o expert_lanes.o lane_sidecar.o ssd_checkpoint.o)
.PHONY: mtp-metadata-cpu
mtp-metadata-cpu: $(BUILD)/test-mtp $(BUILD)/test-gguf-mtp-regression $(BUILD)/tokspur-mtp-metadata
$(BUILD)/test-mtp: tests/test_mtp.cpp $(MTP_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(MTP_OBJECTS) -o "$@"
$(BUILD)/tokspur-mtp-metadata: tests/mtp_metadata_main.cpp $(MTP_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(MTP_OBJECTS) -o "$@"
$(BUILD)/test-gguf-mtp-regression: tests/test_gguf.cpp $(OBJDIR)/gguf.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJDIR)/gguf.o -o "$@"

$(BUILD) $(OBJ_ROOT) $(OBJDIR):
	mkdir -p "$@"

$(OBJDIR)/%.o: src/%.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(OBJDIR)/%.o: src/%.mm $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fobjc-arc -MMD -MP -c "$<" -o "$@"

$(OBJDIR)/carrier_owned_worker.o: src/owned_worker.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(OBJDIR)/carrier_serve_protocol.o: src/serve_protocol.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(filter-out %carrier_owned_worker.o %carrier_serve_protocol.o,$(CARRIER_HTTP_OBJECTS)): $(OBJDIR)/%.o: src/%.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(OBJDIR)/raw_decode_component.o: src/raw_decode_component.cpp include/tokspur/raw_decode.hpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -ffp-contract=off -MMD -MP -c "$<" -o "$@"

$(BUILD)/tokspur-native: $(OBJDIR)/main.o $(ACTIVE_HTTP_OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CXXFLAGS) $(sort $(filter %.o,$^)) $(METAL_FLAGS) -o "$@"

$(BUILD)/scalar-channel-target: $(BUILD)/tokspur-native
	cp -f "$<" "$@"

$(OBJDIR)/server_main.o: server_main.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(OBJDIR)/serve.o: src/serve.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD)/tokspur-server: $(OBJDIR)/server_main.o $(OBJDIR)/serve.o $(OBJDIR)/text.o $(ACTIVE_HTTP_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread $(filter %.o,$^) -framework Foundation -o "$@"

$(addprefix $(BUILD)/test-,$(PATTERN_TESTS)): $(BUILD)/test-%: tests/test_%.cpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) -o "$@"

$(BUILD)/test-pipeline_key: include/tokspur/pipeline_key.hpp
$(BUILD)/test-dflash2_checkpoint: include/tokspur/dflash2_checkpoint.hpp
$(BUILD)/test-derived_pack: tests/grouped_mm_fixture.hpp

$(BUILD)/test-host_telemetry: tests/test_host_telemetry.cpp $(OBJECTS) $(TELEMETRY_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) $(TELEMETRY_OBJECTS) -o "$@"

$(OBJDIR)/host_telemetry_main.o: host_telemetry_main.cpp $(BUILD)/.cflags $(OBJDIR)/.cflags $(OBJECT_POLICY_PREREQUISITE) | $(OBJDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD)/tokspur-host-telemetry: $(OBJDIR)/host_telemetry_main.o $(TELEMETRY_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) -o "$@"

# Base Metal Shaders
$(BUILD)/gemv.air: kernels/gemv.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math $(if $(filter native-plain-loads,$(NATIVE_CONFIG)),-DTOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS=1) -c "$<" -o "$@"

$(BUILD)/graph.air: kernels/graph.metal $(BUILD)/.metalflags $(call metal_policy_prerequisite,$(BUILD)/graph.air) | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"
	@echo '$(TRACKED_METALFLAGS)' > "$@.metalflags"

$(BUILD)/attention.air: kernels/attention.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/q8_many.air: kernels/q8_many.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/q8_split.air: kernels/q8_split.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/gdn_tile.air: kernels/gdn_tile.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -DTOKSPUR_PRIVATE_PREFILL_CHUNK=1 -c "$<" -o "$@"

# Active Carrier Extensions: Exact HC and Wide Q8
$(BUILD)/hc_q8_exact.air: kernels/hc_q8_exact.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/hc_q8_exact_many.air: kernels/hc_q8_exact_many.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/q8_wide_exact.air: kernels/q8_wide_exact.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/q8_wide_many_u4.air: kernels/q8_wide_many_u4.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/hc_rows.air: kernels/hc_rows.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/hc_down_window.air: kernels/hc_down_window.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

# Active Carrier Extensions: QSA Stages and Sparse Attention
$(BUILD)/qsa_score_stage.air: kernels/qsa_score_stage.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/qsa_channel_chronological.air: kernels/qsa_channel_chronological.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/qsa_verify_rows.air: kernels/qsa_verify_rows.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/qsa_sparse_attention.air: kernels/qsa_sparse_attention.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

# Active Carrier Extensions: Dense Prefill and MPP
$(BUILD)/grouped_mm.air: kernels/grouped_mm.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/qsa_prefill.air: kernels/qsa_prefill.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/q8_mpp.air: kernels/q8_mpp.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/grouped_mpp.air: kernels/grouped_mpp.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/grouped_mm_f32.air: kernels/grouped_mm_f32.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/grouped_f32_compact.air: kernels/grouped_mm_f32.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -DTOKSPUR_GROUPED_F32_COMPACT=1 -c "$<" -o "$@"

$(BUILD)/grouped_f32_compensated.air: kernels/grouped_mm_f32.metal experiments/fp32_compensated.h $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -DTOKSPUR_GROUPED_F32_COMPENSATED=1 -c "$<" -o "$@"

$(BUILD)/grouped-f32-compensated.metallib: $(BUILD)/grouped_f32_compensated.air
	xcrun metallib "$<" -o "$@"

$(BUILD)/grouped_f32_reuse.air: kernels/grouped_mm_f32.metal experiments/fp32_compensated.h $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -DTOKSPUR_GROUPED_F32_REUSE=1 -c "$<" -o "$@"

$(BUILD)/grouped-f32-reuse.metallib: $(BUILD)/grouped_f32_reuse.air
	xcrun metallib "$<" -o "$@"

$(BUILD)/grouped-f32-control.metallib: $(BUILD)/grouped_mm_f32.air
	xcrun metallib "$<" -o "$@"

$(BUILD)/grouped-f32-compact.metallib: $(BUILD)/grouped_f32_compact.air
	xcrun metallib "$<" -o "$@"

GROUPED_F32_TEST_DEPS = tests/grouped_f32_test_support.hpp tests/grouped_mm_fixture.hpp tests/moe_projection_reference.hpp tests/raw_decode_reference.hpp include/tokspur/expert_lanes.hpp include/tokspur/numeric.hpp include/tokspur/gguf.hpp include/tokspur/grouped_mm.hpp src/expert_lanes.cpp src/numeric.cpp src/gguf.cpp src/grouped_mm.cpp
$(BUILD)/test-grouped_f32_compact: tests/test_grouped_f32_compact.mm $(GROUPED_F32_TEST_DEPS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -ffp-contract=off -fobjc-arc tests/test_grouped_f32_compact.mm src/expert_lanes.cpp src/numeric.cpp src/gguf.cpp src/grouped_mm.cpp -framework Foundation -framework Metal -o "$@"

$(BUILD)/test-grouped_f32_precision: tests/test_grouped_f32_precision.mm $(GROUPED_F32_TEST_DEPS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -ffp-contract=off -fobjc-arc tests/test_grouped_f32_precision.mm src/expert_lanes.cpp src/numeric.cpp src/gguf.cpp src/grouped_mm.cpp -framework Foundation -framework Metal -o "$@"

$(BUILD)/test-grouped_f32_reuse: tests/test_grouped_f32_reuse.mm $(GROUPED_F32_TEST_DEPS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -ffp-contract=off -fobjc-arc tests/test_grouped_f32_reuse.mm src/expert_lanes.cpp src/numeric.cpp src/gguf.cpp src/grouped_mm.cpp -framework Foundation -framework Metal -o "$@"

.PHONY: grouped-f32-reuse-check
grouped-f32-reuse-check: $(BUILD)/test-grouped_f32_reuse $(BUILD)/grouped-f32-control.metallib $(BUILD)/grouped-f32-compact.metallib $(BUILD)/grouped-f32-compensated.metallib $(BUILD)/grouped-f32-reuse.metallib
	"$(BUILD)/test-grouped_f32_reuse" "$(BUILD)/grouped-f32-control.metallib" "$(BUILD)/grouped-f32-compact.metallib" "$(BUILD)/grouped-f32-compensated.metallib" "$(BUILD)/grouped-f32-reuse.metallib"

.PHONY: grouped-f32-precision-check
grouped-f32-precision-check: $(BUILD)/test-grouped_f32_precision $(BUILD)/grouped-f32-control.metallib $(BUILD)/grouped-f32-compact.metallib $(BUILD)/grouped-f32-compensated.metallib
	"$(BUILD)/test-grouped_f32_precision" "$(BUILD)/grouped-f32-control.metallib" "$(BUILD)/grouped-f32-compact.metallib" "$(BUILD)/grouped-f32-compensated.metallib"

.PHONY: grouped-f32-compact-check
grouped-f32-compact-check: $(BUILD)/test-grouped_f32_compact $(BUILD)/grouped-f32-control.metallib $(BUILD)/grouped-f32-compact.metallib
	"$(BUILD)/test-grouped_f32_compact" "$(BUILD)/grouped-f32-control.metallib" "$(BUILD)/grouped-f32-compact.metallib"

$(BUILD)/dense_q8_staged_f32.air: kernels/dense_q8_staged_f32.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/nax_prefill.air: kernels/nax_prefill.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/dflash2.air: kernels/dflash2.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/planar_moe_v3.air: kernels/planar_moe_v3.metal $(BUILD)/.metalflags $(call metal_policy_prerequisite,$(BUILD)/planar_moe_v3.air) | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"
	@echo '$(TRACKED_METALFLAGS)' > "$@.metalflags"

$(BUILD)/raw_decode.air: kernels/raw_decode.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

# Active Carrier Extensions: MoE Router
$(BUILD)/moe_router_f32_batch.air: kernels/moe_router_f32_batch.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal4.0 -mmacosx-version-min=27.0 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

# Active Carrier Extensions: Recurrence
$(BUILD)/gdn_chunk.air: kernels/gdn_chunk.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/gdn-cs64-reference.air: kernels/gdn_chunk.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -DTOKSPUR_TEST_GDN_CS64_REFERENCE=1 -c "$<" -o "$@"

$(BUILD)/gdn-cs64-reference.metallib: $(BUILD)/gdn-cs64-reference.air
	xcrun metallib "$<" -o "$@"

$(BUILD)/gdn-cs64-reuse.metallib: $(BUILD)/gdn_chunk.air
	xcrun metallib "$<" -o "$@"

$(BUILD)/test-gdn_cs64_reuse: tests/test_gdn_cs64_reuse.mm include/tokspur/gdn_chunk.hpp $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fobjc-arc "$<" -framework Foundation -framework Metal -o "$@"

.PHONY: gdn-cs64-reuse-check
gdn-cs64-reuse-check: $(BUILD)/test-gdn_cs64_reuse $(BUILD)/gdn-cs64-reference.metallib $(BUILD)/gdn-cs64-reuse.metallib
	MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 "$(BUILD)/test-gdn_cs64_reuse" "$(BUILD)/gdn-cs64-reference.metallib" "$(BUILD)/gdn-cs64-reuse.metallib" --validate

$(BUILD)/gdn_legacy_batch.air: kernels/gdn_legacy_batch.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

# Active Carrier Extensions: Verification and Layout
$(BUILD)/mtp_head.air: kernels/mtp_head.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/expert_lanes.air: kernels/expert_lanes.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -DTOKSPUR_PRIVATE_LANE_LAYOUT=1 -c "$<" -o "$@"

$(BUILD)/q8_three_rows.air: kernels/q8_three_rows.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

$(BUILD)/logit_topk.air: kernels/logit_topk.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -ffp-contract=off -c "$<" -o "$@"

$(BUILD)/verify_hc_gdn_fused.air: kernels/verify_hc_gdn_fused.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"

ALL_AIR = $(BUILD)/gemv.air $(BUILD)/graph.air $(BUILD)/attention.air $(BUILD)/q8_many.air $(BUILD)/q8_split.air $(BUILD)/gdn_tile.air \
	$(BUILD)/hc_q8_exact.air $(BUILD)/hc_q8_exact_many.air $(BUILD)/q8_wide_exact.air $(BUILD)/q8_wide_many_u4.air $(BUILD)/hc_rows.air $(BUILD)/hc_down_window.air \
	$(BUILD)/qsa_score_stage.air $(BUILD)/qsa_channel_chronological.air $(BUILD)/qsa_verify_rows.air $(BUILD)/qsa_sparse_attention.air \
	$(BUILD)/grouped_mm.air $(BUILD)/qsa_prefill.air $(BUILD)/q8_mpp.air $(BUILD)/grouped_mpp.air $(BUILD)/grouped_mm_f32.air $(BUILD)/dense_q8_staged_f32.air $(BUILD)/nax_prefill.air $(BUILD)/dflash2.air $(BUILD)/planar_moe_v3.air $(BUILD)/raw_decode.air \
	$(BUILD)/moe_router_f32_batch.air \
	$(BUILD)/gdn_chunk.air $(BUILD)/gdn_legacy_batch.air \
	$(BUILD)/mtp_head.air $(BUILD)/expert_lanes.air $(BUILD)/q8_three_rows.air \
	$(BUILD)/logit_topk.air $(BUILD)/verify_hc_gdn_fused.air

ifeq ($(NATIVE_CONFIG),carrier-diagnostic)
ALL_AIR += $(BUILD)/carrier_routes.air
$(BUILD)/carrier_routes.air: diagnostics/carrier_routes.metal $(BUILD)/.metalflags | $(BUILD)
	xcrun metal $(METALFLAGS) -std=metal3.2 -fno-fast-math -c "$<" -o "$@"
endif

$(BUILD)/tokspur-owned.metallib: Makefile $(ALL_AIR) $(call metal_policy_prerequisite,$(BUILD)/tokspur-owned.metallib) | $(BUILD)
	xcrun metallib $(ALL_AIR) -o "$@"
	@echo '$(TRACKED_METALFLAGS)' > "$@.metalflags"

GPU_TEST_BINS = $(addprefix $(BUILD)/test-,metal graph moe recurrent attention ple stack profile small_gemv block_gemv layout_gemv many_gemv gdn_tile_gpu gdn_core_tile recurrent_tile graph_replay hc_q8_gemv q8_split_gemv)
$(GPU_TEST_BINS): $(BUILD)/test-%: tests/test_%.cpp $(OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJECTS) $(METAL_OBJECTS) $(METAL_FLAGS) -o "$@"

test-metal: $(GPU_TEST_BINS) $(METAL_LIBRARY)
	"$(BUILD)/test-metal" "$(METAL_LIBRARY)"
	"$(BUILD)/test-graph" "$(METAL_LIBRARY)"
	"$(BUILD)/test-moe" "$(METAL_LIBRARY)"
	"$(BUILD)/test-recurrent" "$(METAL_LIBRARY)"
	"$(BUILD)/test-attention" "$(METAL_LIBRARY)"
	"$(BUILD)/test-ple" "$(METAL_LIBRARY)"
	"$(BUILD)/test-stack" "$(METAL_LIBRARY)"
	"$(BUILD)/test-profile" "$(METAL_LIBRARY)"
	"$(BUILD)/test-small_gemv" "$(METAL_LIBRARY)"
	"$(BUILD)/test-block_gemv" "$(METAL_LIBRARY)"
	"$(BUILD)/test-layout_gemv" "$(METAL_LIBRARY)"
	"$(BUILD)/test-many_gemv" "$(METAL_LIBRARY)"
	"$(BUILD)/test-gdn_tile_gpu" "$(METAL_LIBRARY)"
	"$(BUILD)/test-gdn_core_tile" "$(METAL_LIBRARY)"
	"$(BUILD)/test-recurrent_tile" "$(METAL_LIBRARY)"
	"$(BUILD)/test-graph_replay" "$(METAL_LIBRARY)"
	"$(BUILD)/test-hc_q8_gemv" "$(METAL_LIBRARY)"
	"$(BUILD)/test-q8_split_gemv" "$(METAL_LIBRARY)"

CARRIER_GPU_TEST_BINS = $(addprefix $(BUILD)/test-,gdn_ported qsa_sparse_attention qsa_verify_rows_core qsa_prefill chunked_stack prefill_residual hc_rows moe_mm_many_gpu gdn_chunk_gpu q8_three_rows_graph q8_three_rows_stack empirical_challenger_m2_prefill challenger_m2_3 concurrent_dispatch_startup nax_prefill_metal exact_candidates_batch_a exact_candidates_batch_b concurrent_abba_benchmark m4_s3_abba_benchmark challenger_m4_2 challenger_m4_cs64 challenger_m4_3 challenger_m5_2 sampled_stack mtp_head dflash2_context_stage planar_pack_v3 planar_union_profile logit_topk verify_hc_gdn_fused dflash2_shortlist)
ifeq ($(NATIVE_CONFIG),carrier-diagnostic)
CARRIER_GPU_TEST_BINS += $(BUILD)/test-carrier_diagnostic_gpu
endif
$(BUILD)/test-speculative_controls: tests/test_speculative_controls.cpp include/tokspur/speculative_controls.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" -o "$@"
$(CARRIER_GPU_TEST_BINS): $(BUILD)/test-%: tests/test_%.cpp $(OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) $(METAL_OBJECTS) $(METAL_FLAGS) -o "$@"
$(BUILD)/test-runtime_library_exclusion: tests/test_runtime_library_exclusion.cpp $(OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) $(METAL_OBJECTS) $(METAL_FLAGS) -o "$@"
$(BUILD)/test-chunked_stack: tests/gdn_ported_controls.inc tests/qsa_sparse_controls.inc
$(BUILD)/test-challenger_m2_3: tests/test_stack.cpp

$(BUILD)/test-source_retirement: tests/test_source_retirement.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) "$<" -framework Foundation -framework Metal -o "$@"

$(BUILD)/test-planar_union_metadata: tests/test_planar_union_metadata.mm $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fobjc-arc "$<" -framework Foundation -framework Metal -o "$@"

carrier-test-metal: $(CARRIER_GPU_TEST_BINS) $(BUILD)/test-planar_union_metadata $(BUILD)/test-source_retirement $(BUILD)/test-gdn_cs64_reuse $(BUILD)/gdn-cs64-reference.metallib $(METAL_LIBRARY)
	MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 "$(BUILD)/test-gdn_cs64_reuse" "$(BUILD)/gdn-cs64-reference.metallib" "$(METAL_LIBRARY)" --validate
	"$(BUILD)/test-source_retirement" "$(METAL_LIBRARY)" third_party/sources.json
	"$(BUILD)/test-qsa_sparse_attention" "$(METAL_LIBRARY)"
	"$(BUILD)/test-qsa_verify_rows_core" "$(METAL_LIBRARY)"
	"$(BUILD)/test-qsa_prefill" "$(METAL_LIBRARY)"
	"$(BUILD)/test-chunked_stack" "$(METAL_LIBRARY)"
	"$(BUILD)/test-prefill_residual" "$(METAL_LIBRARY)"
	"$(BUILD)/test-gdn_chunk_gpu" --allow-gpu "$(METAL_LIBRARY)" quick
	"$(BUILD)/test-q8_three_rows_graph" --execute-generated "$(METAL_LIBRARY)"
	"$(BUILD)/test-q8_three_rows_stack" --execute-generated "$(METAL_LIBRARY)"
	"$(BUILD)/test-empirical_challenger_m2_prefill" "$(METAL_LIBRARY)"
	"$(BUILD)/test-challenger_m2_3" "$(METAL_LIBRARY)"
	"$(BUILD)/test-planar_pack_v3" "$(METAL_LIBRARY)"
	"$(BUILD)/test-planar_union_profile" "$(METAL_LIBRARY)"
	"$(BUILD)/test-planar_union_metadata" "$(METAL_LIBRARY)"
	"$(BUILD)/test-logit_topk" "$(METAL_LIBRARY)"
	"$(BUILD)/test-mtp_head" "$(METAL_LIBRARY)"
	"$(BUILD)/test-sampled_stack" "$(METAL_LIBRARY)"
	"$(BUILD)/test-sampled_stack" "$(METAL_LIBRARY)" --concurrent
	"$(BUILD)/test-dflash2_context_stage" "$(METAL_LIBRARY)"
	"$(BUILD)/test-verify_hc_gdn_fused" "$(METAL_LIBRARY)"
	"$(BUILD)/test-dflash2_shortlist" "$(METAL_LIBRARY)"

test-challenger_m4_3: $(BUILD)/test-challenger_m4_3 $(METAL_LIBRARY)
	"$(BUILD)/test-challenger_m4_3" "$(METAL_LIBRARY)"

test-concurrent_abba_benchmark: $(BUILD)/test-concurrent_abba_benchmark $(METAL_LIBRARY)
	"$(BUILD)/test-concurrent_abba_benchmark" "$(METAL_LIBRARY)"

test-m4_s3_abba_benchmark: $(BUILD)/test-m4_s3_abba_benchmark $(METAL_LIBRARY)
	"$(BUILD)/test-m4_s3_abba_benchmark" "$(METAL_LIBRARY)"

test: $(TEST_BINS)
	"$(BUILD)/test-literal_markers"
	"$(BUILD)/test-pipeline_key"
	"$(BUILD)/test-dflash2_checkpoint"
	"$(BUILD)/test-shared_prefix_restore"
	"$(BUILD)/test-speculative_controls"
	"$(BUILD)/test-gguf"
	"$(BUILD)/test-numeric"
	"$(BUILD)/test-tokenizer"
	"$(BUILD)/test-unicode"
	"$(BUILD)/test-json"
	"$(BUILD)/test-chat"
	"$(BUILD)/test-operators"
	"$(BUILD)/test-model"
	"$(BUILD)/test-routing"
	"$(BUILD)/test-gdn"
	"$(BUILD)/test-reference"
	"$(BUILD)/test-attention_math"
	"$(BUILD)/test-memory"
	"$(BUILD)/test-shared_prefix_eviction"
	"$(BUILD)/test-sampling"
	"$(BUILD)/test-response"
	"$(BUILD)/test-response_protocol"
	"$(BUILD)/test-q8_layout"
	"$(BUILD)/test-q8_many"
	"$(BUILD)/test-gdn_tile"
	"$(BUILD)/test-recurrent_timing_protocol"
	"$(BUILD)/test-recurrent_replay_timing_protocol"
	"$(BUILD)/test-serve_protocol"
	"$(BUILD)/test-http"
	"$(BUILD)/test-owned_worker"
	"$(BUILD)/test-prompt_lookup"
	"$(BUILD)/test-transcript_replay"
	"$(BUILD)/test-derived_pack"
	"$(BUILD)/test-host_telemetry"
	"$(BUILD)/test-ssd_checkpoint"

sanitize:
	$(MAKE) test BUILD="$(SANITIZE_BUILD)" ENABLE_METAL="$(ENABLE_METAL)" CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -fno-omit-frame-pointer'
	$(MAKE) ci-components BUILD="$(CI_SANITIZE_BUILD)" ENABLE_METAL="$(ENABLE_METAL)" CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -fsanitize-address-use-after-scope -fno-omit-frame-pointer'

$(BUILD)/fuzz-gguf: tests/fuzz_gguf.cpp src/gguf.cpp include/tokspur/gguf.hpp $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -DTOKSPUR_STANDALONE_FUZZ=1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/fuzz_gguf.cpp src/gguf.cpp -o "$@"

fuzz: $(BUILD)/test-gguf $(BUILD)/fuzz-gguf
	"$(BUILD)/test-gguf" --write-fuzz-seeds "$(BUILD)/fuzz-corpus"
	"$(BUILD)/fuzz-gguf" "$(BUILD)/fuzz-corpus"

# Optional: Apple's installed toolchain may not include the libFuzzer archive.
$(BUILD)/coverage-fuzz-gguf: tests/fuzz_gguf.cpp src/gguf.cpp include/tokspur/gguf.hpp $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer tests/fuzz_gguf.cpp src/gguf.cpp -o "$@"

coverage-fuzz: $(BUILD)/test-gguf $(BUILD)/coverage-fuzz-gguf
	"$(BUILD)/test-gguf" --write-fuzz-seeds "$(BUILD)/fuzz-corpus"
	"$(BUILD)/coverage-fuzz-gguf" "$(BUILD)/fuzz-corpus" -runs=50000 -max_len=65536 -timeout=5

# Model-free CI component contract suites (replacing obsolete .mk overlays)
CI_COMPONENT_TESTS = prepared_sampling carrier_diagnostic measured_controller nax_split_policy http_concurrent mtp_wire mtp_sampling mtp_anchor_state mtp_packet64 http_fast_math http_gdn_ported moe_mm_many q8_three_rows probe_pack_census pack_v2_offset_safety derived_pack moe_adversarial_m2 sparse_residual_dual_order shared_kv_checkpoint ssd_checkpoint
CI_COMPONENT_BINS = $(addprefix $(BUILD)/test-,$(CI_COMPONENT_TESTS))
ci-components: $(CI_COMPONENT_BINS)
	"$(BUILD)/test-carrier_diagnostic"
	"$(BUILD)/test-measured_controller"
	"$(BUILD)/test-nax_split_policy"
	"$(BUILD)/test-http_concurrent"
	"$(BUILD)/test-mtp_wire"
	"$(BUILD)/test-mtp_sampling"
	"$(BUILD)/test-prepared_sampling"
	"$(BUILD)/test-mtp_anchor_state"
	"$(BUILD)/test-mtp_packet64"
	"$(BUILD)/test-http_fast_math"
	"$(BUILD)/test-http_gdn_ported"
	"$(BUILD)/test-moe_mm_many"
	"$(BUILD)/test-q8_three_rows"
	"$(BUILD)/test-probe_pack_census"
	"$(BUILD)/test-pack_v2_offset_safety"
	"$(BUILD)/test-derived_pack"
	"$(BUILD)/test-moe_adversarial_m2"
	"$(BUILD)/test-sparse_residual_dual_order"
	"$(BUILD)/test-shared_kv_checkpoint"
	"$(BUILD)/test-ssd_checkpoint"

test-sparse_residual_dual_order: $(BUILD)/test-sparse_residual_dual_order
	"$(BUILD)/test-sparse_residual_dual_order"

$(BUILD)/test-sparse_residual_dual_order: tests/test_sparse_residual_dual_order.cpp $(OBJDIR)/sampling.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) tests/test_sparse_residual_dual_order.cpp $(OBJDIR)/sampling.o -o "$@"

$(BUILD)/test-pack_v2_offset_safety: tests/test_pack_v2_offset_safety.cpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) -o "$@"

$(BUILD)/test-probe_pack_census: ../tests/probe_pack_census_challenge.cpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJECTS) -o "$@"

$(BUILD)/test-model_pack_fallback_challenge: ../tests/test_model_pack_fallback_challenge.cpp $(OBJECTS) $(OBJDIR)/mtp.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJECTS) $(OBJDIR)/mtp.o -o "$@"

$(BUILD)/test-http_concurrent: tests/test_http_concurrent.cpp $(CARRIER_ALL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(CARRIER_ALL_OBJECTS) -o "$@"

$(BUILD)/test-mtp_wire: tests/test_mtp_wire.cpp $(CARRIER_ALL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(CARRIER_ALL_OBJECTS) -o "$@"

$(BUILD)/test-mtp_sampling: tests/test_mtp_sampling.cpp $(OBJDIR)/mtp_sampling.o $(OBJDIR)/sampled_path.o $(OBJDIR)/sampling.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJDIR)/mtp_sampling.o $(OBJDIR)/sampled_path.o $(OBJDIR)/sampling.o -o "$@"
$(BUILD)/test-prepared_sampling: tests/test_prepared_sampling.cpp $(OBJDIR)/mtp_sampling.o $(OBJDIR)/sampled_path.o $(OBJDIR)/sampling.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" $(OBJDIR)/mtp_sampling.o $(OBJDIR)/sampled_path.o $(OBJDIR)/sampling.o -o "$@"


$(BUILD)/test-mtp_anchor_state: tests/test_mtp_anchor_state.cpp include/tokspur/mtp_anchor_state.hpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) -o "$@"

$(BUILD)/test-mtp_packet64: tests/test_mtp_packet64.cpp $(CARRIER_ALL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(CARRIER_ALL_OBJECTS) -o "$@"

$(BUILD)/test-http_fast_math: tests/test_http_fast_math.cpp tests/test_http_branch8.cpp tests/test_http_verify8.cpp $(ACTIVE_HTTP_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(ACTIVE_HTTP_OBJECTS) -o "$@"

$(BUILD)/test-http_gdn_ported: tests/test_http_gdn_ported.cpp tests/test_http_qsa_sparse.cpp tests/test_http_prefill_hc_down_window.cpp tests/test_http_prefill_router_window.cpp tests/test_http_prefill_hc_rows.cpp tests/test_http_prefill512.cpp tests/test_mtp_wire.cpp $(CARRIER_ALL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(CARRIER_ALL_OBJECTS) -o "$@"

$(BUILD)/test-moe_mm_many: tests/test_moe_mm_many.cpp src/moe_mm_many_plan.cpp src/grouped_mm.cpp $(OBJDIR)/gguf.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) tests/test_moe_mm_many.cpp src/moe_mm_many_plan.cpp src/grouped_mm.cpp $(OBJDIR)/gguf.o -o "$@"

$(BUILD)/test-moe_adversarial_m2: tests/test_moe_adversarial_m2.cpp src/moe_mm_many_plan.cpp src/grouped_mm.cpp $(OBJDIR)/gguf.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) tests/test_moe_adversarial_m2.cpp src/moe_mm_many_plan.cpp src/grouped_mm.cpp $(OBJDIR)/gguf.o -o "$@"

$(BUILD)/test-q8_three_rows: tests/test_q8_three_rows.cpp src/q8_three_rows.cpp $(OBJDIR)/q8_many.o $(OBJDIR)/gguf.o $(OBJDIR)/numeric.o $(OBJDIR)/expert_lanes.o $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CARRIER_CPPFLAGS) $(CXXFLAGS) -fno-fast-math tests/test_q8_three_rows.cpp src/q8_three_rows.cpp $(OBJDIR)/q8_many.o $(OBJDIR)/gguf.o $(OBJDIR)/numeric.o $(OBJDIR)/expert_lanes.o -o "$@"

$(BUILD)/test-shared_kv_checkpoint: tests/test_shared_kv_checkpoint.cpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) -o "$@"

$(BUILD)/test-ssd_checkpoint: tests/test_ssd_checkpoint.cpp $(OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) -o "$@"

ifeq ($(ENABLE_METAL),1)
$(BUILD)/tokspur-build-pack-v3: tools/build_pack_v3.cpp $(OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) $(METAL_OBJECTS) $(METAL_FLAGS) -o "$@"
endif

# Explicit historical http-fast-math configuration selector (0144a14 feature boundary).
http-fast-math:
	$(MAKE) NATIVE_CONFIG=http-fast-math BUILD="$(HTTP_FAST_MATH_BUILD)" "$(HTTP_FAST_MATH_BUILD)/tokspur-server" "$(HTTP_FAST_MATH_BUILD)/test-http_fast_math"

$(BUILD)/test-native_plain_loads: tests/test_native_plain_loads.cpp $(OBJECTS) $(METAL_OBJECTS) $(BUILD)/.cflags | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread "$<" $(OBJECTS) $(METAL_OBJECTS) $(METAL_FLAGS) -o "$@"

native-plain-loads-component:
	$(MAKE) NATIVE_CONFIG=native-plain-loads BUILD=../build/native-plain-loads-component ../build/native-plain-loads-component/test-native_plain_loads ../build/native-plain-loads-component/tokspur-owned.metallib

rust-check:
	$(MAKE) -f rust.mk rust-check

-include $(wildcard $(OBJDIR)/*.d)

clean:
	rm -rf $(BUILD)/obj $(BUILD)/*.air $(BUILD)/*.metallib $(BUILD)/*.air.metalflags $(BUILD)/tokspur-* $(BUILD)/scalar-channel-target $(BUILD)/test-* $(BUILD)/fuzz-* $(BUILD)/.cflags $(BUILD)/.metalflags "$(SANITIZE_BUILD)" "$(CI_SANITIZE_BUILD)" "$(HTTP_FAST_MATH_BUILD)" ../build/native-plain-loads-component

# Deliberately excluded from all, serving libraries, staging and package manifests.
PROBE_BUILD ?= ../build/runtime-check-probe
.PHONY: runtime-check-probe
runtime-check-probe: $(PROBE_BUILD)/probe $(PROBE_BUILD)/tokspur-runtime-check-probe.metallib
$(PROBE_BUILD):
	mkdir -p "$@"
$(PROBE_BUILD)/probe: tests/runtime_check_probe.mm | $(PROBE_BUILD)
	$(CXX) -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -fobjc-arc "$<" -framework Foundation -framework Metal -o "$@"
$(PROBE_BUILD)/runtime_checks.air: probes/runtime_checks.metal | $(PROBE_BUILD)
	xcrun metal -std=metal3.2 -fno-fast-math -c "$<" -o "$@"
$(PROBE_BUILD)/tokspur-runtime-check-probe.metallib: $(PROBE_BUILD)/runtime_checks.air
	xcrun metallib "$<" -o "$@"

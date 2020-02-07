#pragma once

using CHUNK = Chunk<GeneralizedChunk<CHUNK_SIZE>,CHUNK_SIZE, SMALLEST_PAGE_SIZE>;
using PAGECHUNK = PageChunk<GeneralizedChunk<CHUNK_SIZE>,CHUNK_SIZE>;

using CQ = OuroborosChunks<ChunkQueue, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using PQ = OuroborosPages<PageQueue, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VLCQ = OuroborosChunks<ChunkQueueVL, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VLPQ = OuroborosPages<PageQueueVL, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VACQ = OuroborosChunks<ChunkQueueVA, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VAPQ = OuroborosPages<PageQueueVA, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;

using OuroCQ = Ouroboros<OuroborosChunks<ChunkQueue, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroPQ = Ouroboros<OuroborosPages<PageQueue, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVLCQ = Ouroboros<OuroborosChunks<ChunkQueueVL, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVLPQ = Ouroboros<OuroborosPages<PageQueueVL, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVACQ = Ouroboros<OuroborosChunks<ChunkQueueVA, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVAPQ = Ouroboros<OuroborosPages<PageQueueVA, GeneralizedChunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;

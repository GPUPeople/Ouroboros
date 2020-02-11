#pragma once

// using CHUNK = ChunkIndexChunk<Chunk<CHUNK_SIZE>,CHUNK_SIZE, SMALLEST_PAGE_SIZE>;
// using PAGECHUNK = PageChunk<Chunk<CHUNK_SIZE>,CHUNK_SIZE>;

using CQ = OuroborosChunks<ChunkQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using PQ = OuroborosPages<PageQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VLCQ = OuroborosChunks<ChunkQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VLPQ = OuroborosPages<PageQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VACQ = OuroborosChunks<ChunkQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;
using VAPQ = OuroborosPages<PageQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>;

using OuroCQ = Ouroboros<OuroborosChunks<ChunkQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroPQ = Ouroboros<OuroborosPages<PageQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVLCQ = Ouroboros<OuroborosChunks<ChunkQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVLPQ = Ouroboros<OuroborosPages<PageQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVACQ = Ouroboros<OuroborosChunks<ChunkQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;
using OuroVAPQ = Ouroboros<OuroborosPages<PageQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>>;

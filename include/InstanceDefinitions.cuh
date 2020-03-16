#pragma once

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

using MultiOuroCQ = Ouroboros<
	OuroborosChunks<ChunkQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosChunks<ChunkQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;
using MultiOuroPQ = Ouroboros<
	OuroborosPages<PageQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosPages<PageQueue, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;
using MultiOuroVLCQ = Ouroboros<
	OuroborosChunks<ChunkQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosChunks<ChunkQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;
using MultiOuroVLPQ = Ouroboros<
	OuroborosPages<PageQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosPages<PageQueueVL, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;
using MultiOuroVACQ = Ouroboros<
	OuroborosChunks<ChunkQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosChunks<ChunkQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;
using MultiOuroVAPQ = Ouroboros<
	OuroborosPages<PageQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE, NUM_QUEUES>,
	OuroborosPages<PageQueueVA, Chunk<CHUNK_SIZE>, SMALLEST_PAGE_SIZE << NUM_QUEUES, 2>>;

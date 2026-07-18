"""Nemori quickstart example."""
import asyncio

from dotenv import load_dotenv

from nemori import NemoriMemory, MemoryConfig

load_dotenv()


async def main():
    # Credentials, endpoints, and the embedding model are resolved from the
    # environment. Keep the buffer below its auto-processing threshold so the
    # explicit flush below is deterministic.
    config = MemoryConfig(buffer_size_min=10)

    async with NemoriMemory(config=config) as memory:
        health = await memory.health()
        print(f"System healthy: {health.healthy}")

        await memory.add_messages("alice", [
            {"role": "user", "content": "I just moved to Tokyo last month"},
            {"role": "assistant", "content": "How exciting! How are you finding life in Tokyo?"},
            {"role": "user", "content": "Love it! The food is amazing, especially ramen"},
        ])

        episodes = await memory.flush("alice")
        print(f"Created {len(episodes)} episodes")

        results = await memory.search("alice", "Where does Alice live?")
        print(results)


if __name__ == "__main__":
    asyncio.run(main())

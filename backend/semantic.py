"""
Semantic distance and bridge-finding logic.
This is where the magic happens - finding the most interesting connections.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations, product
from dataclasses import dataclass


@dataclass
class KnowledgeBite:
    """A piece of knowledge from a participant."""
    participant_id: str
    participant_name: str
    text: str
    embedding: List[float] = None


@dataclass  
class Connection:
    """A discovered connection between knowledge bites."""
    bite1: KnowledgeBite
    bite2: KnowledgeBite
    distance: float
    bridge_concept: str = None  # For bridge mode


@dataclass
class GroupConnection:
    """A discovered group connection across multiple bites."""
    members: List[KnowledgeBite]
    score: float
    strategy: str
    details: Optional[Dict[str, Any]] = None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Calculate cosine distance (1 - similarity)."""
    return 1 - cosine_similarity(a, b)


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    if not embeddings:
        return []
    arr = np.array(embeddings)
    return arr.mean(axis=0).tolist()


def distance_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(values)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean())
    }


def find_maximum_distance_pairs(
    bites: List[KnowledgeBite],
    cross_participant_only: bool = True,
    top_k: int = 3,
    debug: Optional[Dict[str, Any]] = None
) -> List[Connection]:
    """
    Find the pairs of knowledge bites that are furthest apart semantically,
    but still within a reasonable range (not complete noise).
    
    This is the core "reverse RAG" idea - instead of finding closest matches,
    we find the most distant ones that still have some thread of connection.
    """
    connections = []
    threshold_min = 0.2
    threshold_max = 1.5
    total_pairs = 0
    cross_pairs = 0
    with_embeddings = 0
    in_range = 0
    all_distances: List[float] = []
    in_range_distances: List[float] = []
    candidate_pairs: List[Dict[str, Any]] = []
    
    for i, bite1 in enumerate(bites):
        for j, bite2 in enumerate(bites):
            if i >= j:
                continue
            total_pairs += 1
            if cross_participant_only and bite1.participant_id == bite2.participant_id:
                continue
            cross_pairs += 1
            if bite1.embedding is None or bite2.embedding is None:
                continue
            with_embeddings += 1
                
            dist = cosine_distance(bite1.embedding, bite2.embedding)
            all_distances.append(dist)
            
            # We want distant but not completely unrelated
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # Sweet spot is probably 0.3 - 0.8 for "distant but connected"
            if threshold_min < dist < threshold_max:
                in_range += 1
                in_range_distances.append(dist)
                connections.append(Connection(
                    bite1=bite1,
                    bite2=bite2,
                    distance=dist
                ))
                candidate_pairs.append({
                    "participant1": bite1.participant_name,
                    "bite1": bite1.text,
                    "participant2": bite2.participant_name,
                    "bite2": bite2.text,
                    "distance": dist
                })
    
    # Sort by distance descending (we want the MOST distant)
    connections.sort(key=lambda c: c.distance, reverse=True)
    
    if debug is not None:
        def stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"min": 0.0, "max": 0.0, "mean": 0.0}
            arr = np.array(values)
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean())
            }

        candidate_pairs.sort(key=lambda c: c["distance"], reverse=True)
        debug.update({
            "strategy": "max_distance",
            "cross_participant_only": cross_participant_only,
            "threshold": {"min": threshold_min, "max": threshold_max},
            "pair_counts": {
                "total_pairs": total_pairs,
                "cross_participant_pairs": cross_pairs,
                "with_embeddings": with_embeddings,
                "within_threshold": in_range
            },
            "distance_stats": {
                "all_pairs": stats(all_distances),
                "within_threshold": stats(in_range_distances)
            },
            "candidate_pairs": candidate_pairs[:10]
        })

    return connections[:top_k]


def find_bridge_connections(
    bites: List[KnowledgeBite],
    bridge_concepts: List[Tuple[str, List[float]]],
    cross_participant_only: bool = True,
    top_k: int = 3
) -> List[Connection]:
    """
    Find pairs of bites that are distant from each other but both close to
    some unexpected third concept (the bridge).
    
    bridge_concepts: List of (concept_name, embedding) tuples
    """
    connections = []
    
    for i, bite1 in enumerate(bites):
        for j, bite2 in enumerate(bites):
            if i >= j:
                continue
            if cross_participant_only and bite1.participant_id == bite2.participant_id:
                continue
            if bite1.embedding is None or bite2.embedding is None:
                continue
            
            # Distance between the two bites
            bite_distance = cosine_distance(bite1.embedding, bite2.embedding)
            
            # Only consider if bites are sufficiently distant
            if bite_distance < 0.3:
                continue
            
            # Find if there's a bridge concept close to both
            for concept_name, concept_embedding in bridge_concepts:
                dist1 = cosine_distance(bite1.embedding, concept_embedding)
                dist2 = cosine_distance(bite2.embedding, concept_embedding)
                
                # Both should be reasonably close to the bridge
                if dist1 < 0.5 and dist2 < 0.5:
                    # Score: high bite_distance + low bridge distances = good
                    score = bite_distance - (dist1 + dist2) / 2
                    connections.append(Connection(
                        bite1=bite1,
                        bite2=bite2,
                        distance=score,
                        bridge_concept=concept_name
                    ))
    
    connections.sort(key=lambda c: c.distance, reverse=True)
    return connections[:top_k]


def find_asymmetric_gifts(
    giver_bites: List[KnowledgeBite],
    receiver_bites: List[KnowledgeBite],
    top_k: int = 3,
    debug: Optional[Dict[str, Any]] = None
) -> List[Connection]:
    """
    For couples/intimate mode: find something from the giver that
    unexpectedly illuminates something about the receiver.
    
    We look for giver bites that are moderately distant from receiver bites
    but have some surprising resonance.
    """
    connections = []
    threshold_min = 0.25
    threshold_max = 0.7
    total_pairs = 0
    with_embeddings = 0
    in_range = 0
    all_distances: List[float] = []
    in_range_distances: List[float] = []
    candidate_pairs: List[Dict[str, Any]] = []
    
    for giver_bite in giver_bites:
        for receiver_bite in receiver_bites:
            total_pairs += 1
            if giver_bite.embedding is None or receiver_bite.embedding is None:
                continue
            with_embeddings += 1
            
            dist = cosine_distance(giver_bite.embedding, receiver_bite.embedding)
            all_distances.append(dist)
            
            # Sweet spot: not too close (obvious) not too far (random)
            if threshold_min < dist < threshold_max:
                in_range += 1
                in_range_distances.append(dist)
                connections.append(Connection(
                    bite1=giver_bite,
                    bite2=receiver_bite,
                    distance=dist
                ))
                candidate_pairs.append({
                    "participant1": giver_bite.participant_name,
                    "bite1": giver_bite.text,
                    "participant2": receiver_bite.participant_name,
                    "bite2": receiver_bite.text,
                    "distance": dist
                })
    
    # Sort by distance - we want the ones in the sweet spot
    # Prefer slightly higher distances for more surprise
    connections.sort(key=lambda c: c.distance, reverse=True)
    
    if debug is not None:
        def stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"min": 0.0, "max": 0.0, "mean": 0.0}
            arr = np.array(values)
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean())
            }

        candidate_pairs.sort(key=lambda c: c["distance"], reverse=True)
        debug.update({
            "strategy": "asymmetric_gift",
            "threshold": {"min": threshold_min, "max": threshold_max},
            "pair_counts": {
                "total_pairs": total_pairs,
                "with_embeddings": with_embeddings,
                "within_threshold": in_range
            },
            "distance_stats": {
                "all_pairs": stats(all_distances),
                "within_threshold": stats(in_range_distances)
            },
            "candidate_pairs": candidate_pairs[:10]
        })

    return connections[:top_k]


def find_triplet_connections(
    bites: List[KnowledgeBite],
    top_k: int = 3,
    debug: Optional[Dict[str, Any]] = None
) -> List[GroupConnection]:
    """
    Find triplets of bites from different participants where all pairwise
    distances are in the "distant but connected" range.
    """
    threshold_min = 0.2
    threshold_max = 1.5
    triplets: List[GroupConnection] = []
    total_triplets = 0
    in_range = 0
    scores: List[float] = []
    candidate_groups: List[Dict[str, Any]] = []

    for i, j, k in combinations(range(len(bites)), 3):
        bite1 = bites[i]
        bite2 = bites[j]
        bite3 = bites[k]
        if len({bite1.participant_id, bite2.participant_id, bite3.participant_id}) < 3:
            continue
        if bite1.embedding is None or bite2.embedding is None or bite3.embedding is None:
            continue
        total_triplets += 1
        d12 = cosine_distance(bite1.embedding, bite2.embedding)
        d13 = cosine_distance(bite1.embedding, bite3.embedding)
        d23 = cosine_distance(bite2.embedding, bite3.embedding)
        if all(threshold_min < d < threshold_max for d in (d12, d13, d23)):
            score = min(d12, d13, d23)
            in_range += 1
            scores.append(score)
            triplets.append(GroupConnection(
                members=[bite1, bite2, bite3],
                score=score,
                strategy="triplet_weave",
                details={"pair_distances": [d12, d13, d23]}
            ))
            candidate_groups.append({
                "members": [
                    {"participant": bite1.participant_name, "bite": bite1.text},
                    {"participant": bite2.participant_name, "bite": bite2.text},
                    {"participant": bite3.participant_name, "bite": bite3.text}
                ],
                "score": score,
                "pair_distances": [d12, d13, d23]
            })

    triplets.sort(key=lambda g: g.score, reverse=True)

    if debug is not None:
        candidate_groups.sort(key=lambda g: g["score"], reverse=True)
        debug.update({
            "strategy": "triplet_weave",
            "threshold": {"min": threshold_min, "max": threshold_max},
            "triplet_counts": {
                "total_triplets": total_triplets,
                "within_threshold": in_range
            },
            "score_stats": distance_stats(scores),
            "candidate_groups": candidate_groups[:5]
        })

    return triplets[:top_k]


def find_centroid_constellation(
    bites: List[KnowledgeBite],
    top_k: int = 1,
    debug: Optional[Dict[str, Any]] = None
) -> List[GroupConnection]:
    """
    Find a group that includes all participants by choosing one bite per participant
    that sits near the shared centroid, then maximize pairwise distance.
    """
    participants: Dict[str, List[KnowledgeBite]] = {}
    for bite in bites:
        participants.setdefault(bite.participant_id, []).append(bite)

    if len(participants) < 2:
        return []

    all_embeddings = [bite.embedding for bite in bites if bite.embedding is not None]
    centroid = compute_centroid(all_embeddings)
    if not centroid:
        return []

    centroid_distances: Dict[str, List[Tuple[KnowledgeBite, float]]] = {}
    all_centroid_distances: List[float] = []
    for participant_id, p_bites in participants.items():
        distances = []
        for bite in p_bites:
            if bite.embedding is None:
                continue
            dist = cosine_distance(bite.embedding, centroid)
            distances.append((bite, dist))
            all_centroid_distances.append(dist)
        centroid_distances[participant_id] = distances

    target = float(np.mean(all_centroid_distances)) if all_centroid_distances else 0.5
    candidate_map: Dict[str, List[KnowledgeBite]] = {}
    for participant_id, distances in centroid_distances.items():
        distances.sort(key=lambda bd: abs(bd[1] - target))
        candidates = [bd[0] for bd in distances[:2]]
        if not candidates:
            continue
        candidate_map[participant_id] = candidates

    if len(candidate_map) < 2:
        return []

    participant_ids = list(candidate_map.keys())
    candidate_lists = [candidate_map[pid] for pid in participant_ids]
    groups: List[GroupConnection] = []
    candidate_groups: List[Dict[str, Any]] = []

    for combo in product(*candidate_lists):
        pair_distances = []
        valid = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                dist = cosine_distance(combo[i].embedding, combo[j].embedding)
                pair_distances.append(dist)
                if dist < 0.15:
                    valid = False
        if not valid:
            continue
        score = float(np.mean(pair_distances)) if pair_distances else 0.0
        groups.append(GroupConnection(
            members=list(combo),
            score=score,
            strategy="centroid_constellation",
            details={"pair_distances": pair_distances}
        ))
        candidate_groups.append({
            "members": [
                {"participant": member.participant_name, "bite": member.text}
                for member in combo
            ],
            "score": score
        })

    groups.sort(key=lambda g: g.score, reverse=True)

    if debug is not None:
        candidate_groups.sort(key=lambda g: g["score"], reverse=True)
        debug.update({
            "strategy": "centroid_constellation",
            "centroid_distance_stats": distance_stats(all_centroid_distances),
            "candidate_groups": candidate_groups[:3]
        })

    return groups[:top_k]


def find_bridge_chain(
    bites: List[KnowledgeBite],
    top_k: int = 1,
    debug: Optional[Dict[str, Any]] = None
) -> List[GroupConnection]:
    """
    Build a chain that includes all participants by starting from the most distant
    pair and iteratively adding the next most distant-but-connected bite.
    """
    threshold_min = 0.2
    threshold_max = 1.5
    participants = {bite.participant_id for bite in bites}
    if len(participants) < 2:
        return []

    # Precompute distance matrix
    n = len(bites)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bites[i].embedding is None or bites[j].embedding is None:
                continue
            dist = cosine_distance(bites[i].embedding, bites[j].embedding)
            distances[i][j] = dist
            distances[j][i] = dist

    # Start with the most distant cross-participant pair
    best_pair = None
    best_dist = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if bites[i].participant_id == bites[j].participant_id:
                continue
            dist = distances[i][j]
            if threshold_min < dist < threshold_max and dist > best_dist:
                best_dist = dist
                best_pair = (i, j)

    if best_pair is None:
        for i in range(n):
            for j in range(i + 1, n):
                if bites[i].participant_id == bites[j].participant_id:
                    continue
                dist = distances[i][j]
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (i, j)

    if best_pair is None:
        return []

    chain_indices = [best_pair[0], best_pair[1]]
    chain_members = [bites[best_pair[0]], bites[best_pair[1]]]
    used_participants = {bites[best_pair[0]].participant_id, bites[best_pair[1]].participant_id}
    chain_steps = [{
        "from": bites[best_pair[0]].participant_name,
        "to": bites[best_pair[1]].participant_name,
        "distance": best_dist
    }]

    while len(used_participants) < len(participants):
        best_candidate = None
        best_score = -1.0
        best_link = None

        for idx, bite in enumerate(bites):
            if bite.participant_id in used_participants:
                continue
            distances_to_chain = [distances[idx][existing] for existing in chain_indices]
            if not distances_to_chain:
                continue
            min_dist = min(distances_to_chain)
            if threshold_min < min_dist < threshold_max and min_dist > best_score:
                best_score = min_dist
                best_candidate = idx
                nearest_index = chain_indices[distances_to_chain.index(min_dist)]
                best_link = (nearest_index, idx, min_dist)

        if best_candidate is None:
            # Fallback: choose the closest candidate to the chain
            fallback_best = None
            fallback_dist = float("inf")
            fallback_link = None
            for idx, bite in enumerate(bites):
                if bite.participant_id in used_participants:
                    continue
                distances_to_chain = [distances[idx][existing] for existing in chain_indices]
                if not distances_to_chain:
                    continue
                min_dist = min(distances_to_chain)
                if min_dist < fallback_dist:
                    fallback_dist = min_dist
                    fallback_best = idx
                    nearest_index = chain_indices[distances_to_chain.index(min_dist)]
                    fallback_link = (nearest_index, idx, min_dist)
            best_candidate = fallback_best
            best_link = fallback_link
            best_score = fallback_dist if fallback_dist != float("inf") else 0.0

        if best_candidate is None:
            break

        chain_indices.append(best_candidate)
        chain_members.append(bites[best_candidate])
        used_participants.add(bites[best_candidate].participant_id)
        if best_link:
            chain_steps.append({
                "from": bites[best_link[0]].participant_name,
                "to": bites[best_link[1]].participant_name,
                "distance": best_link[2]
            })

    score = float(np.mean([step["distance"] for step in chain_steps])) if chain_steps else 0.0
    group = GroupConnection(
        members=chain_members,
        score=score,
        strategy="bridge_chain",
        details={"chain": chain_steps}
    )

    if debug is not None:
        debug.update({
            "strategy": "bridge_chain",
            "threshold": {"min": threshold_min, "max": threshold_max},
            "steps": chain_steps
        })

    return [group][:top_k]


def prepare_prompt_for_context(
    connections: List[Connection],
    context: str,
    mode: str
) -> str:
    """
    Create the prompt for the LLM based on context and discovered connections.
    """
    connection_texts = []
    for i, conn in enumerate(connections):
        connection_texts.append(
            f"Connection {i+1}:\n"
            f"  - From {conn.bite1.participant_name}: \"{conn.bite1.text}\"\n"
            f"  - From {conn.bite2.participant_name}: \"{conn.bite2.text}\"\n"
            f"  - Semantic distance: {conn.distance:.3f}"
            + (f"\n  - Bridge concept: {conn.bridge_concept}" if conn.bridge_concept else "")
        )
    
    connections_str = "\n\n".join(connection_texts)
    
    response_format = (
        "Return strictly valid JSON with keys \"ideas\" and \"reasoning\".\n"
        "- ideas: plain text only (no markdown, no asterisks); format as three numbered lines like \"1) ...\".\n"
        "- reasoning: 1-3 short sentences of high-level rationale, no step-by-step reasoning, plain text only.\n"
    )

    if context == "team":
        return f"""You are helping a team brainstorm by finding unexpected connections between their ideas.

Here are the most surprising semantic connections found between team members' knowledge bites:

{connections_str}

Based on these distant-but-connected ideas, generate:
1. A surprising insight or innovation that bridges these concepts
2. A concrete "What if..." question the team could explore
3. An unexpected angle or approach this suggests for their work

Be specific, actionable, and delightfully unexpected. Show them something they couldn't have seen alone.

{response_format}"""

    elif context == "strangers":
        return f"""You are creating magical conversation starters for strangers who just met.

Here are the most surprising semantic connections found between their shared knowledge bites:

{connections_str}

Create a sense of serendipity - like the universe conspired to bring these people together.
Generate:
1. A whimsical observation about what connects them
2. A playful question they could explore together  
3. A tiny collaborative "micro-adventure" they could do right now

Be warm, curious, and slightly magical. Give them permission to be playful with each other.

{response_format}"""

    elif context == "couples":
        return f"""You are creating intimate moments of discovery for a couple.

Here are the unexpected resonances found between their knowledge bites:

{connections_str}

Create tender "I didn't know that about you" moments.
Generate:
1. An observation about what their hidden connection reveals
2. A gentle question one could ask the other
3. A small ritual or moment they could share based on this discovery

Be warm, intimate, and treat their connection as precious. Help them see each other anew.

{response_format}"""

    else:
        return f"""Here are some surprising connections between knowledge bites:

{connections_str}

What unexpected story, insight, or question emerges from these connections?

{response_format}"""


def prepare_prompt_for_group_context(
    groups: List[GroupConnection],
    context: str,
    mode: str
) -> str:
    group_texts = []
    for i, group in enumerate(groups):
        member_lines = []
        for member in group.members:
            member_lines.append(
                f"  - {member.participant_name}: \"{member.text}\""
            )
        group_texts.append(
            f"Group {i+1} (score: {group.score:.3f}, strategy: {group.strategy}):\n"
            + "\n".join(member_lines)
        )

    groups_str = "\n\n".join(group_texts)
    response_format = (
        "Return strictly valid JSON with keys \"ideas\" and \"reasoning\".\n"
        "- ideas: plain text only (no markdown, no asterisks); format as three numbered lines like \"1) ...\".\n"
        "- reasoning: 1-3 short sentences of high-level rationale, no step-by-step reasoning, plain text only.\n"
    )

    if context == "team":
        return f"""You are helping a team brainstorm by finding unexpected connections that include everyone.

Here are the most interesting group connections found across participants:

{groups_str}

Based on these connections, generate:
1. A surprising insight or innovation that bridges the full group
2. A concrete "What if..." question the team could explore
3. An unexpected angle or approach this suggests for their work

Be specific, actionable, and delightfully unexpected. Show them something they couldn't have seen alone.

{response_format}"""

    if context == "strangers":
        return f"""You are creating magical conversation starters for strangers who just met.

Here are group connections that include everyone:

{groups_str}

Create a sense of serendipity - like the universe conspired to bring these people together.
Generate:
1. A whimsical observation about what connects them
2. A playful question they could explore together
3. A tiny collaborative "micro-adventure" they could do right now

Be warm, curious, and slightly magical. Give them permission to be playful with each other.

{response_format}"""

    if context == "couples":
        return f"""You are creating intimate moments of discovery for a couple.

Here are group connections that include both partners:

{groups_str}

Create tender "I didn't know that about you" moments.
Generate:
1. An observation about what their hidden connection reveals
2. A gentle question one could ask the other
3. A small ritual or moment they could share based on this discovery

Be warm, intimate, and treat their connection as precious. Help them see each other anew.

{response_format}"""

    return f"""Here are some surprising group connections:

{groups_str}

What unexpected story, insight, or question emerges from these connections?

{response_format}"""

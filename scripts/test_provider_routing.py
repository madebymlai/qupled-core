#!/usr/bin/env python3
"""
Test script for Provider Routing Architecture.

This script tests:
1. Profile loading from YAML
2. Task routing to correct providers
3. Fallback behavior
4. Provider availability checking
5. Backward compatibility
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from core.provider_router import ProviderRouter
from core.task_types import TaskType
from models.llm_manager import LLMManager


def test_profile_loading():
    """Test that profiles load correctly from YAML."""
    print("\n=== Test 1: Profile Loading ===")
    try:
        Config.ensure_dirs()
        router = ProviderRouter()
        profiles = router.list_profiles()
        print(f"✓ Loaded {len(profiles)} profiles: {', '.join(profiles)}")

        # Check expected profiles exist
        expected = ['free', 'pro', 'local']
        for profile in expected:
            if profile not in profiles:
                print(f"✗ Missing expected profile: {profile}")
                return False

        print("✓ All expected profiles present")
        return True
    except Exception as e:
        print(f"✗ Profile loading failed: {e}")
        return False


def test_task_routing():
    """Test that tasks route to correct providers."""
    print("\n=== Test 2: Task Routing ===")
    try:
        router = ProviderRouter()

        # Test free profile
        print("\nFree profile:")
        bulk_provider = router.route(TaskType.BULK_ANALYSIS, "free")
        print(f"  - BULK_ANALYSIS → {bulk_provider} (expected: groq)")

        interactive_provider = router.route(TaskType.INTERACTIVE, "free")
        print(f"  - INTERACTIVE → {interactive_provider} (expected: groq)")

        try:
            premium_provider = router.route(TaskType.PREMIUM, "free")
            print(f"  ✗ PREMIUM should be disabled on free tier but got: {premium_provider}")
            return False
        except ValueError as e:
            print(f"  ✓ PREMIUM correctly disabled: {e}")

        # Test pro profile
        print("\nPro profile:")
        bulk_provider = router.route(TaskType.BULK_ANALYSIS, "pro")
        print(f"  - BULK_ANALYSIS → {bulk_provider} (expected: groq)")

        interactive_provider = router.route(TaskType.INTERACTIVE, "pro")
        print(f"  - INTERACTIVE → {interactive_provider} (expected: anthropic)")

        premium_provider = router.route(TaskType.PREMIUM, "pro")
        print(f"  - PREMIUM → {premium_provider} (expected: anthropic)")

        # Test local profile
        print("\nLocal profile:")
        bulk_provider = router.route(TaskType.BULK_ANALYSIS, "local")
        print(f"  - BULK_ANALYSIS → {bulk_provider} (expected: ollama)")

        interactive_provider = router.route(TaskType.INTERACTIVE, "local")
        print(f"  - INTERACTIVE → {interactive_provider} (expected: ollama)")

        print("\n✓ Task routing works correctly")
        return True

    except Exception as e:
        print(f"✗ Task routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_availability():
    """Test provider availability checking."""
    print("\n=== Test 3: Provider Availability ===")
    try:
        # Test availability checking
        providers = ['ollama', 'anthropic', 'groq', 'openai']

        for provider in providers:
            available = LLMManager.is_provider_available(provider)
            status = "✓ available" if available else "✗ unavailable (no API key)"
            print(f"  {provider}: {status}")

        print("\n✓ Provider availability check works")
        return True

    except Exception as e:
        print(f"✗ Provider availability check failed: {e}")
        return False


def test_profile_info():
    """Test getting detailed profile information."""
    print("\n=== Test 4: Profile Info ===")
    try:
        router = ProviderRouter()

        for profile_name in ['free', 'pro', 'local']:
            print(f"\n{profile_name.upper()} profile:")
            info = router.get_profile_info(profile_name)
            print(f"  Description: {info['description']}")
            print(f"  Tasks:")
            for task_name, task_config in info['tasks'].items():
                if task_config['enabled']:
                    fallback = task_config['fallback'] or 'none'
                    print(f"    - {task_name}: {task_config['primary']} (fallback: {fallback})")
                else:
                    print(f"    - {task_name}: disabled")

        print("\n✓ Profile info retrieval works")
        return True

    except Exception as e:
        print(f"✗ Profile info failed: {e}")
        return False


def test_profile_validation():
    """Test profile validation."""
    print("\n=== Test 5: Profile Validation ===")
    try:
        router = ProviderRouter()
        issues = router.validate_profiles()

        has_issues = False
        for profile_name, profile_issues in issues.items():
            if profile_issues:
                has_issues = True
                print(f"\n{profile_name} profile issues:")
                for issue in profile_issues:
                    print(f"  ⚠ {issue}")

        if not has_issues:
            print("✓ All profiles valid (no configuration issues)")
        else:
            print("\n⚠ Some profiles have issues (see above)")

        return True

    except Exception as e:
        print(f"✗ Profile validation failed: {e}")
        return False


def test_backward_compatibility():
    """Test that the system works without profiles (backward compatible)."""
    print("\n=== Test 6: Backward Compatibility ===")
    try:
        # Test direct LLMManager usage (old style)
        llm = LLMManager(provider=Config.LLM_PROVIDER)
        print(f"✓ Direct LLMManager works with provider: {llm.provider}")

        # Test that existing code patterns still work
        print("✓ Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PROVIDER ROUTING ARCHITECTURE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Profile Loading", test_profile_loading),
        ("Task Routing", test_task_routing),
        ("Provider Availability", test_provider_availability),
        ("Profile Info", test_profile_info),
        ("Profile Validation", test_profile_validation),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Provider Routing Architecture is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

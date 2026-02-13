(function () {
  "use strict";

  angular
    .module("guardrailsApp", [])
    .service("Storage", Storage)
    .service("Api", Api)
    .controller("ChatController", ChatController);

  function Storage($window) {
    var hasLocalStorage = false;
    try {
      var k = "__storage_test__";
      $window.localStorage.setItem(k, "1");
      $window.localStorage.removeItem(k);
      hasLocalStorage = true;
    } catch (e) {
      hasLocalStorage = false;
    }

    this.get = function (key, fallback) {
      if (!hasLocalStorage) return fallback;
      var value = $window.localStorage.getItem(key);
      return value === null ? fallback : value;
    };

    this.set = function (key, value) {
      if (!hasLocalStorage) return;
      if (value === null || value === undefined) {
        $window.localStorage.removeItem(key);
      } else {
        $window.localStorage.setItem(key, String(value));
      }
    };
  }

  function Api($http) {
    this.getInfo = function () {
      return $http.get("/api/info", { timeout: 8000 }).then(function (r) {
        return r.data;
      });
    };

    this.getHealth = function () {
      return $http.get("/health", { timeout: 8000 }).then(function (r) {
        return r.data;
      });
    };

    this.sendChat = function (endpoint, message, conversationId, apiKey, headerName) {
      var headers = { "Content-Type": "application/json" };
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;

      var payload = { message: message };
      if (conversationId) payload.conversation_id = conversationId;

      return $http
        .post(endpoint, payload, { headers: headers, timeout: 60000 })
        .then(function (r) {
          return r.data;
        });
    };

    this.checkInputSafety = function (message, apiKey, headerName) {
      var headers = { "Content-Type": "application/json" };
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;
      return $http
        .post("/api/tools/check_input_safety", { message: message }, { headers: headers, timeout: 15000 })
        .then(function (r) {
          return r.data;
        });
    };

    this.checkOutputSafety = function (message, apiKey, headerName) {
      var headers = { "Content-Type": "application/json" };
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;
      return $http
        .post("/api/tools/check_output_safety", { message: message }, { headers: headers, timeout: 15000 })
        .then(function (r) {
          return r.data;
        });
    };

    this.createConversation = function (apiKey, headerName) {
      var headers = { "Content-Type": "application/json" };
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;
      return $http.post("/api/conversations", {}, { headers: headers, timeout: 8000 }).then(function (r) {
        return r.data;
      });
    };

    this.getConversation = function (conversationId, apiKey, headerName) {
      var headers = {};
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;
      return $http
        .get("/api/conversations/" + encodeURIComponent(conversationId), { headers: headers, timeout: 8000 })
        .then(function (r) {
          return r.data;
        });
    };

    this.deleteConversation = function (conversationId, apiKey, headerName) {
      var headers = {};
      if (apiKey) headers[headerName || "X-API-Key"] = apiKey;
      return $http
        .delete("/api/conversations/" + encodeURIComponent(conversationId), { headers: headers, timeout: 8000 })
        .then(function (r) {
          return r.data;
        });
    };
  }

  function ChatController($timeout, $window, Storage, Api) {
    var vm = this;

    var STORAGE_API_KEY = "nemo_guardrails_api_key";

    vm.serverBase = $window.location.origin;
    vm.mode = "live";
    vm.sending = false;

    vm.apiKeyRequired = true;
    vm.headerName = "X-API-Key";
    vm.apiKey = Storage.get(STORAGE_API_KEY, "");

    vm.connection = { ok: false, message: "Connecting..." };
    vm.health = { ok: false, guardrailsLoaded: false };

    vm.conversationId = null;
    vm.draft = "";
    vm.messages = [];

    vm.tools = {
      inputText: "",
      inputBusy: false,
      inputResult: null,
      outputText: "",
      outputBusy: false,
      outputResult: null,
    };

    vm.setMode = function (mode) {
      vm.mode = mode === "mock" ? "mock" : "live";
    };

    vm.persistApiKey = function () {
      Storage.set(STORAGE_API_KEY, vm.apiKey || "");
    };

    vm.useExample = function (text) {
      vm.draft = text;
      focusComposer();
    };

    vm.clearMessages = function () {
      vm.messages = [];
      scrollToBottom();
    };

    vm.newConversation = function () {
      vm.conversationId = null;
      vm.clearMessages();

      var key = (vm.apiKey || "").trim();
      if (vm.apiKeyRequired && !key) {
        vm.addSystem("New conversation started.");
        focusComposer();
        return;
      }

      Api.createConversation(key, vm.headerName)
        .then(function (data) {
          vm.conversationId = data.conversation_id || null;
          if (vm.conversationId) {
            vm.addSystem("New conversation created: " + vm.conversationId);
          } else {
            vm.addSystem("New conversation started.");
          }
        })
        .catch(function () {
          vm.addSystem("New conversation started.");
        })
        .finally(function () {
          focusComposer();
        });
    };

    vm.copyConversationId = function () {
      if (!vm.conversationId) return;
      var value = vm.conversationId;

      if ($window.navigator && $window.navigator.clipboard && $window.navigator.clipboard.writeText) {
        $window.navigator.clipboard
          .writeText(value)
          .then(function () {
            vm.addSystem("Conversation id copied.");
          })
          .catch(function () {
            fallbackCopy(value);
          });
        return;
      }

      fallbackCopy(value);
    };

    vm.onKeydown = function (ev) {
      if (!ev) return;
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        vm.send();
      }
    };

    vm.addSystem = function (text) {
      vm.messages.push({
        role: "system",
        content: text,
        time: new Date(),
        guardrailsTriggered: false,
      });
      scrollToBottom();
    };

    vm.send = function () {
      var message = (vm.draft || "").trim();
      if (!message || vm.sending) return;

      if (vm.apiKeyRequired && !(vm.apiKey || "").trim()) {
        vm.connection = { ok: false, message: "Missing API key. Check server logs." };
        vm.addSystem("API key required. Paste it in the panel.");
        return;
      }

      vm.sending = true;
      vm.draft = "";

      vm.messages.push({ role: "user", content: message, time: new Date(), guardrailsTriggered: false });
      scrollToBottom();

      var endpoint = vm.mode === "mock" ? "/api/chat/test" : "/api/chat";

      Api.sendChat(endpoint, message, vm.conversationId, (vm.apiKey || "").trim(), vm.headerName)
        .then(function (data) {
          vm.conversationId = data.conversation_id || vm.conversationId;

          vm.messages.push({
            role: "assistant",
            content: data.response || "",
            time: new Date(),
            guardrailsTriggered: !!data.guardrails_triggered,
          });
          vm.connection = { ok: true, message: "Ready" };
        })
        .catch(function (err) {
          var status = (err && err.status) || 0;
          var detail = (err && err.data && err.data.detail) || "Request failed";

          if (status === 401 || status === 403) {
            vm.connection = { ok: false, message: "Auth failed" };
            vm.messages.push({
              role: "assistant",
              content: "Authentication failed. Check your API key.",
              time: new Date(),
              guardrailsTriggered: false,
            });
          } else if (status === 429) {
            vm.connection = { ok: false, message: "Rate limited" };
            vm.messages.push({
              role: "assistant",
              content: "Rate limit exceeded. " + detail,
              time: new Date(),
              guardrailsTriggered: false,
            });
          } else if (status === 503) {
            vm.connection = { ok: false, message: "Rails not initialized" };
            vm.messages.push({
              role: "assistant",
              content: "Guardrails are not initialized yet. Try again in a moment.",
              time: new Date(),
              guardrailsTriggered: false,
            });
          } else {
            vm.connection = { ok: false, message: "Error" };
            vm.messages.push({
              role: "assistant",
              content: detail,
              time: new Date(),
              guardrailsTriggered: false,
            });
          }
        })
        .finally(function () {
          vm.sending = false;
          scrollToBottom();
          focusComposer();
        });
    };

    function scrollToBottom() {
      $timeout(function () {
        var el = $window.document.getElementById("messages");
        if (el) el.scrollTop = el.scrollHeight;
      }, 0);
    }

    function focusComposer() {
      $timeout(function () {
        var el = $window.document.querySelector(".composer__input");
        if (el) el.focus();
      }, 0);
    }

    function fallbackCopy(value) {
      try {
        var input = $window.document.createElement("input");
        input.value = value;
        input.style.position = "fixed";
        input.style.left = "-9999px";
        $window.document.body.appendChild(input);
        input.select();
        $window.document.execCommand("copy");
        $window.document.body.removeChild(input);
        vm.addSystem("Conversation id copied.");
      } catch (e) {
        vm.addSystem("Copy failed.");
      }
    }

    function refreshStatus() {
      Api.getInfo()
        .then(function (info) {
          vm.apiKeyRequired = !!info.api_key_required;
          vm.headerName = info.header_name || "X-API-Key";
        })
        .catch(function () {
          vm.apiKeyRequired = true;
        });

      Api.getHealth()
        .then(function (health) {
          vm.health = {
            ok: health.status === "healthy",
            guardrailsLoaded: !!health.guardrails_loaded,
          };
          vm.connection = {
            ok: true,
            message: vm.health.guardrailsLoaded ? "Ready" : "Server up (rails not loaded)",
          };
        })
        .catch(function () {
          vm.health = { ok: false, guardrailsLoaded: false };
          vm.connection = { ok: false, message: "Cannot reach server" };
        });
    }

    refreshStatus();
    $timeout(refreshStatus, 2000);
    focusComposer();

    vm.checkInputSafety = function () {
      var msg = (vm.tools.inputText || "").trim();
      if (!msg || vm.tools.inputBusy) return;

      var key = (vm.apiKey || "").trim();
      if (vm.apiKeyRequired && !key) {
        vm.tools.inputResult = { is_safe: false, details: ["API key required"] };
        return;
      }

      vm.tools.inputBusy = true;
      vm.tools.inputResult = null;
      Api.checkInputSafety(msg, key, vm.headerName)
        .then(function (data) {
          vm.tools.inputResult = data;
        })
        .catch(function (err) {
          vm.tools.inputResult = {
            is_safe: false,
            details: [((err && err.data && err.data.detail) || "Check failed")],
          };
        })
        .finally(function () {
          vm.tools.inputBusy = false;
        });
    };

    vm.checkOutputSafety = function () {
      var msg = (vm.tools.outputText || "").trim();
      if (!msg || vm.tools.outputBusy) return;

      var key = (vm.apiKey || "").trim();
      if (vm.apiKeyRequired && !key) {
        vm.tools.outputResult = { is_safe: false, details: ["API key required"] };
        return;
      }

      vm.tools.outputBusy = true;
      vm.tools.outputResult = null;
      Api.checkOutputSafety(msg, key, vm.headerName)
        .then(function (data) {
          vm.tools.outputResult = data;
        })
        .catch(function (err) {
          vm.tools.outputResult = {
            is_safe: false,
            details: [((err && err.data && err.data.detail) || "Check failed")],
          };
        })
        .finally(function () {
          vm.tools.outputBusy = false;
        });
    };

    vm.exportConversation = function () {
      var payload = {
        exported_at: new Date().toISOString(),
        conversation_id: vm.conversationId,
        mode: vm.mode,
        messages: vm.messages,
      };

      try {
        var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        var url = $window.URL.createObjectURL(blob);
        var a = $window.document.createElement("a");
        a.href = url;
        a.download = "guardrails-conversation-" + (vm.conversationId || "new") + ".json";
        $window.document.body.appendChild(a);
        a.click();
        $window.document.body.removeChild(a);
        $window.URL.revokeObjectURL(url);
      } catch (e) {
        vm.addSystem("Export failed.");
      }
    };
  }
})();
